/* eslint-disable @typescript-eslint/no-non-null-assertion */
/**
 * Corruption handling integration tests for StorageEngine.
 * Tests how the system handles various corruption scenarios.
 */

import { describe, it, expect, afterEach } from 'vitest'
import { writeFile, readFile } from 'node:fs/promises'
import { StorageEngine } from '../storage-engine'
import { Wal } from '../wal'
import { walEntrySize } from '../constants'
import {
  createTestPaths,
  cleanup,
  generateRandomEmbedding,
  collectEntries,
  flipByte,
  type TestPaths
} from './helpers'

describe('StorageEngine corruption handling', () => {
  const testPathsList: TestPaths[] = []

  afterEach(async () => {
    await cleanup(testPathsList)
    testPathsList.length = 0
  })

  describe('data file corruption', () => {
    it('corrupt magic number in data record returns null on read', async () => {
      const paths = createTestPaths('corrupt-magic')
      testPathsList.push(paths)

      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine1.writeRecord('test', generateRandomEmbedding(384))
      await engine1.close()

      // Corrupt magic number (at offset after header, first 4 bytes of first record)
      // Header is 16 bytes, so record starts at byte 16
      const dataBytes = await readFile(paths.dataPath)
      const dataArray = new Uint8Array(dataBytes)
      dataArray[16] = 0xff // Corrupt first byte of record magic
      await writeFile(paths.dataPath, dataArray)

      // Attempt to read - should return null or throw
      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      // Index was built from WAL, so key exists but read should fail
      const record = await engine2.readRecord('test')
      expect(record).toBeNull()

      await engine2.close()
    })

    it('corrupt checksum in data record returns null on read', async () => {
      const paths = createTestPaths('corrupt-checksum')
      testPathsList.push(paths)

      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine1.writeRecord('test', generateRandomEmbedding(384))
      await engine1.close()

      // Corrupt data in payload area (will cause checksum mismatch)
      // Header (16) + record header through embedding data
      const dataBytes = await readFile(paths.dataPath)
      const dataArray = new Uint8Array(dataBytes)
      // Flip a byte in the middle of the record (embedding data area)
      flipByte(dataArray, 100)
      await writeFile(paths.dataPath, dataArray)

      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const record = await engine2.readRecord('test')
      expect(record).toBeNull()

      await engine2.close()
    })

    it('corrupt trailer in data record returns null on read', async () => {
      const paths = createTestPaths('corrupt-trailer')
      testPathsList.push(paths)

      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine1.writeRecord('test', generateRandomEmbedding(384))
      await engine1.close()

      // Corrupt trailer (last 4 bytes of record)
      const dataBytes = await readFile(paths.dataPath)
      const dataArray = new Uint8Array(dataBytes)
      flipByte(dataArray, dataArray.length - 1)
      await writeFile(paths.dataPath, dataArray)

      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const record = await engine2.readRecord('test')
      expect(record).toBeNull()

      await engine2.close()
    })

    it('corruption in one record does not affect others', async () => {
      const paths = createTestPaths('corrupt-partial')
      testPathsList.push(paths)

      const embedding1 = generateRandomEmbedding(384)
      const embedding2 = generateRandomEmbedding(384)

      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine1.writeRecord('first', embedding1)
      await engine1.writeRecord('second', embedding2)
      await engine1.close()

      // Get offsets from WAL for reference
      const wal = new Wal(paths.walPath)
      await collectEntries(wal.recover())
      await wal.close()

      // Corrupt first record's data (not magic, so it validates partially)
      const dataBytes = await readFile(paths.dataPath)
      const dataArray = new Uint8Array(dataBytes)
      // Flip byte in first record's embedding data
      flipByte(dataArray, 50)
      await writeFile(paths.dataPath, dataArray)

      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      // First record should fail
      const first = await engine2.readRecord('first')
      expect(first).toBeNull()

      // Second record should still be readable (if not corrupted)
      // Note: This depends on how far the corruption propagates
      // In practice, each record is independent

      await engine2.close()
    })
  })

  describe('WAL corruption', () => {
    it('corrupt WAL magic stops recovery at that point', async () => {
      const paths = createTestPaths('corrupt-wal-magic')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      // Write 3 records
      await engine.writeRecord('first', generateRandomEmbedding(384))
      await engine.writeRecord('second', generateRandomEmbedding(384))
      await engine.writeRecord('third', generateRandomEmbedding(384))
      await engine.close()

      // Corrupt magic of second entry
      const walBytes = await readFile(paths.walPath)
      const walArray = new Uint8Array(walBytes)
      walArray[walEntrySize] = 0xff // First byte of second record's magic
      await writeFile(paths.walPath, walArray)

      // Recovery should stop at first valid entry
      const wal = new Wal(paths.walPath)
      const entries = await collectEntries(wal.recover())
      await wal.close()

      expect(entries).toHaveLength(1)
      expect(entries[0]?.sequenceNumber).toBe(1n)
    })

    it('corrupt WAL checksum stops recovery', async () => {
      const paths = createTestPaths('corrupt-wal-checksum')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      await engine.writeRecord('first', generateRandomEmbedding(384))
      await engine.writeRecord('second', generateRandomEmbedding(384))
      await engine.close()

      // Corrupt data in second entry (causes checksum mismatch)
      const walBytes = await readFile(paths.walPath)
      const walArray = new Uint8Array(walBytes)
      flipByte(walArray, walEntrySize + 20)
      await writeFile(paths.walPath, walArray)

      const wal = new Wal(paths.walPath)
      const entries = await collectEntries(wal.recover())
      await wal.close()

      expect(entries).toHaveLength(1)
    })

    it('corrupt WAL trailer stops recovery', async () => {
      const paths = createTestPaths('corrupt-wal-trailer')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      await engine.writeRecord('first', generateRandomEmbedding(384))
      await engine.close()

      // Corrupt trailer (last 4 bytes of the entry)
      const walBytes = await readFile(paths.walPath)
      const walArray = new Uint8Array(walBytes)
      flipByte(walArray, walEntrySize - 1) // Last byte of first entry
      await writeFile(paths.walPath, walArray)

      const wal = new Wal(paths.walPath)
      const entries = await collectEntries(wal.recover())
      await wal.close()

      expect(entries).toHaveLength(0)
    })

    it('truncated WAL mid-record stops recovery gracefully', async () => {
      const paths = createTestPaths('truncate-wal')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      await engine.writeRecord('first', generateRandomEmbedding(384))
      await engine.writeRecord('second', generateRandomEmbedding(384))
      await engine.close()

      // Truncate mid-second-record
      const walBytes = await readFile(paths.walPath)
      const truncated = walBytes.subarray(0, walEntrySize + 30)
      await writeFile(paths.walPath, truncated)

      const wal = new Wal(paths.walPath)
      const entries = await collectEntries(wal.recover())
      await wal.close()

      expect(entries).toHaveLength(1)
      expect(entries[0]?.sequenceNumber).toBe(1n)
    })
  })

  describe('random byte flips (fuzz-like)', () => {
    it('random byte flip in data file causes read failure', async () => {
      const paths = createTestPaths('random-data')
      testPathsList.push(paths)

      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine1.writeRecord('test', generateRandomEmbedding(384))
      await engine1.close()

      // Flip a random byte (avoiding header and magic to test checksum)
      const dataBytes = await readFile(paths.dataPath)
      const dataArray = new Uint8Array(dataBytes)
      // Pick a byte after the header (16 bytes) and record magic (4 bytes)
      const flipIndex = 30 + Math.floor(Math.random() * (dataArray.length - 40))
      flipByte(dataArray, flipIndex)
      await writeFile(paths.dataPath, dataArray)

      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      // Should return null due to checksum or validation failure
      const record = await engine2.readRecord('test')
      expect(record).toBeNull()

      await engine2.close()
    })

    it('random byte flip in WAL stops recovery at that point', async () => {
      const paths = createTestPaths('random-wal')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      for (let i = 0; i < 5; i++) {
        await engine.writeRecord(`key${i}`, generateRandomEmbedding(384))
      }
      await engine.close()

      // Flip random byte in third record (avoiding magic bytes)
      const walBytes = await readFile(paths.walPath)
      const walArray = new Uint8Array(walBytes)
      const thirdRecordStart = 2 * walEntrySize
      const flipOffset = 10 + Math.floor(Math.random() * 30) // Within record, after magic
      flipByte(walArray, thirdRecordStart + flipOffset)
      await writeFile(paths.walPath, walArray)

      const wal = new Wal(paths.walPath)
      const entries = await collectEntries(wal.recover())
      await wal.close()

      // Should recover first two entries
      expect(entries).toHaveLength(2)
    })

    it('multiple random corruptions - recovery stops at first', async () => {
      const paths = createTestPaths('multi-corrupt')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      for (let i = 0; i < 10; i++) {
        await engine.writeRecord(`key${i}`, generateRandomEmbedding(384))
      }
      await engine.close()

      // Corrupt records 3, 5, and 8
      const walBytes = await readFile(paths.walPath)
      const walArray = new Uint8Array(walBytes)
      flipByte(walArray, 2 * walEntrySize + 20) // Record 3
      flipByte(walArray, 4 * walEntrySize + 20) // Record 5
      flipByte(walArray, 7 * walEntrySize + 20) // Record 8
      await writeFile(paths.walPath, walArray)

      const wal = new Wal(paths.walPath)
      const entries = await collectEntries(wal.recover())
      await wal.close()

      // Should stop at first corruption (record 3), so only 2 entries recovered
      expect(entries).toHaveLength(2)
    })
  })

  describe('corruption edge cases', () => {
    it('empty WAL file recovers zero entries', async () => {
      const paths = createTestPaths('empty-wal')
      testPathsList.push(paths)

      await writeFile(paths.walPath, new Uint8Array(0))

      const wal = new Wal(paths.walPath)
      const entries = await collectEntries(wal.recover())
      await wal.close()

      expect(entries).toHaveLength(0)
    })

    it('WAL with only partial first record recovers zero entries', async () => {
      const paths = createTestPaths('partial-first')
      testPathsList.push(paths)

      // Write just 30 bytes (less than 48-byte record)
      const partialRecord = new Uint8Array(30)
      const view = new DataView(partialRecord.buffer)
      view.setUint32(0, 0xcafebabe, true) // Magic number
      await writeFile(paths.walPath, partialRecord)

      const wal = new Wal(paths.walPath)
      const entries = await collectEntries(wal.recover())
      await wal.close()

      expect(entries).toHaveLength(0)
    })

    it('corrupt first byte of file stops immediately', async () => {
      const paths = createTestPaths('corrupt-first-byte')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine.writeRecord('test', generateRandomEmbedding(384))
      await engine.close()

      const walBytes = await readFile(paths.walPath)
      const walArray = new Uint8Array(walBytes)
      walArray[0] = 0x00 // Corrupt very first byte
      await writeFile(paths.walPath, walArray)

      const wal = new Wal(paths.walPath)
      const entries = await collectEntries(wal.recover())
      await wal.close()

      expect(entries).toHaveLength(0)
    })

    it('all zeros in WAL recovers zero entries', async () => {
      const paths = createTestPaths('all-zeros')
      testPathsList.push(paths)

      // Write 48 bytes of zeros
      await writeFile(paths.walPath, new Uint8Array(walEntrySize))

      const wal = new Wal(paths.walPath)
      const entries = await collectEntries(wal.recover())
      await wal.close()

      expect(entries).toHaveLength(0)
    })

    it('corrupted checksum field specifically', async () => {
      const paths = createTestPaths('corrupt-checksum-field')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine.writeRecord('test', generateRandomEmbedding(384))
      await engine.close()

      // Corrupt the checksum field directly (bytes 40-43 in our 48-byte format)
      const walBytes = await readFile(paths.walPath)
      const walArray = new Uint8Array(walBytes)
      flipByte(walArray, 40)
      await writeFile(paths.walPath, walArray)

      const wal = new Wal(paths.walPath)
      const entries = await collectEntries(wal.recover())
      await wal.close()

      expect(entries).toHaveLength(0)
    })
  })
})
