/* eslint-disable @typescript-eslint/no-non-null-assertion */
/**
 * Crash simulation integration tests for StorageEngine.
 * Tests various crash scenarios and data recovery.
 */

import { describe, it, expect, afterEach } from 'vitest'
import { stat, writeFile, readFile } from 'node:fs/promises'
import { StorageEngine } from '../storage-engine'
import { Wal } from '../wal'
import { serializeDataRecord } from '../data-format'
import { opType, walEntrySize } from '../constants'
import {
  createTestPaths,
  cleanup,
  generateRandomEmbedding,
  collectEntries,
  embeddingsEqual,
  type TestPaths
} from './helpers'

describe('StorageEngine crash simulation', () => {
  const testPathsList: TestPaths[] = []

  afterEach(async () => {
    await cleanup(testPathsList)
    testPathsList.length = 0
  })

  describe('crash after data write but before WAL write (orphaned data)', () => {
    it('orphaned data is ignored on recovery', async () => {
      const paths = createTestPaths('crash-orphan')
      testPathsList.push(paths)

      const embedding = generateRandomEmbedding(384)

      // Write data to data file WITHOUT corresponding WAL entry
      const serialized = serializeDataRecord({
        opType: opType.insert,
        sequenceNumber: 1n,
        timestamp: BigInt(Date.now()),
        key: 'orphan',
        dimension: 384,
        embedding
      })

      await writeFile(paths.dataPath, serialized)

      // Create empty WAL (no entries)
      await writeFile(paths.walPath, new Uint8Array(0))

      // Recover - should find nothing (orphaned data ignored)
      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      expect(engine.count()).toBe(0)
      expect(await engine.readRecord('orphan')).toBeNull()

      await engine.close()
    })

    it('valid data with WAL is recovered alongside orphaned data', async () => {
      const paths = createTestPaths('crash-partial')
      testPathsList.push(paths)

      const validEmbedding = generateRandomEmbedding(384)

      // First, create a valid database with one embedding
      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine1.writeRecord('valid', validEmbedding)
      await engine1.close()

      // Get the current data file size
      const originalSize = (await stat(paths.dataPath)).size

      // Append orphaned data to the data file (simulates crash after data write)
      const orphanData = serializeDataRecord({
        opType: opType.insert,
        sequenceNumber: 999n,
        timestamp: BigInt(Date.now()),
        key: 'orphan',
        dimension: 384,
        embedding: generateRandomEmbedding(384)
      })

      const existingData = await readFile(paths.dataPath)
      const combined = new Uint8Array(existingData.length + orphanData.length)
      combined.set(existingData, 0)
      combined.set(orphanData, existingData.length)
      await writeFile(paths.dataPath, combined)

      // Verify data file grew
      const newSize = (await stat(paths.dataPath)).size
      expect(newSize).toBeGreaterThan(originalSize)

      // Recover - should only find the valid embedding
      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      expect(engine2.count()).toBe(1)
      expect(engine2.hasKey('valid')).toBe(true)
      expect(engine2.hasKey('orphan')).toBe(false)

      const record = await engine2.readRecord('valid')
      expect(record).not.toBeNull()
      expect(embeddingsEqual(record!.embedding, validEmbedding)).toBe(true)

      await engine2.close()
    })
  })

  describe('crash during WAL write (partial 48-byte record)', () => {
    it('partial WAL record is ignored on recovery', async () => {
      const paths = createTestPaths('crash-partial-wal')
      testPathsList.push(paths)

      // Write one valid embedding
      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine1.writeRecord('valid', generateRandomEmbedding(384))
      await engine1.close()

      // Append partial (truncated) WAL record - simulates crash mid-write
      const partialRecord = new Uint8Array(30) // Less than 48 bytes
      const view = new DataView(partialRecord.buffer)
      view.setUint32(0, 0xcafebabe, true) // Write magic number at start

      const existingWal = await readFile(paths.walPath)
      const combined = new Uint8Array(existingWal.length + partialRecord.length)
      combined.set(existingWal, 0)
      combined.set(partialRecord, existingWal.length)
      await writeFile(paths.walPath, combined)

      // Recover - should only get the first valid entry
      const wal = new Wal(paths.walPath)
      const recoveredEntries = await collectEntries(wal.recover())

      expect(recoveredEntries).toHaveLength(1)
      expect(recoveredEntries[0]?.sequenceNumber).toBe(1n)

      await wal.close()
    })

    it('truncation at various points within WAL record', async () => {
      const truncationPoints = [4, 10, 20, 30, 40, 47] // Various offsets < 48

      for (const truncateAt of truncationPoints) {
        const paths = createTestPaths(`crash-trunc-${truncateAt}`)
        testPathsList.push(paths)

        // Write two valid entries
        const engine = await StorageEngine.create({
          dataPath: paths.dataPath,
          dimension: 384
        })
        await engine.writeRecord('first', generateRandomEmbedding(384))
        await engine.writeRecord('second', generateRandomEmbedding(384))
        await engine.close()

        // Truncate second record at specific point
        const walData = await readFile(paths.walPath)
        const truncated = walData.subarray(0, walEntrySize + truncateAt)
        await writeFile(paths.walPath, truncated)

        // Recover - should only get first entry
        const wal = new Wal(paths.walPath)
        const entries = await collectEntries(wal.recover())

        expect(entries).toHaveLength(1)
        expect(entries[0]?.sequenceNumber).toBe(1n)

        await wal.close()
      }
    })
  })

  describe('crash after WAL write (should recover perfectly)', () => {
    it('committed WAL entry is fully recoverable', async () => {
      const paths = createTestPaths('crash-after-wal')
      testPathsList.push(paths)

      const embedding = generateRandomEmbedding(384)

      // Create database and insert embedding (full write path)
      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine1.writeRecord('test', embedding)
      await engine1.close()

      // Verify files exist with expected sizes
      const dataStats = await stat(paths.dataPath)
      const walStats = await stat(paths.walPath)
      expect(dataStats.size).toBeGreaterThan(0)
      expect(walStats.size).toBe(walEntrySize)

      // Recover (simulate restart)
      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const recovered = await engine2.readRecord('test')
      expect(recovered).not.toBeNull()
      expect(embeddingsEqual(recovered!.embedding, embedding)).toBe(true)

      await engine2.close()
    })

    it('sequence of operations recovers completely', async () => {
      const paths = createTestPaths('crash-sequence')
      testPathsList.push(paths)

      const embedding1 = generateRandomEmbedding(384)
      const embedding2 = generateRandomEmbedding(384)
      const embedding3 = generateRandomEmbedding(384)
      const embedding1Updated = generateRandomEmbedding(384)

      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      // Insert, update, delete sequence
      await engine1.writeRecord('alice', embedding1)
      await engine1.writeRecord('bob', embedding2)
      await engine1.writeRecord('alice', embedding1Updated) // Update
      await engine1.deleteRecord('bob')
      await engine1.writeRecord('charlie', embedding3)

      await engine1.close()

      // Verify WAL has 5 entries (3 inserts/updates + 1 delete + 1 insert)
      const walStats = await stat(paths.walPath)
      expect(walStats.size).toBe(walEntrySize * 5)

      // Recover
      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const alice = await engine2.readRecord('alice')
      const bob = await engine2.readRecord('bob')
      const charlie = await engine2.readRecord('charlie')

      expect(alice).not.toBeNull()
      expect(embeddingsEqual(alice!.embedding, embedding1Updated)).toBe(true) // Updated
      expect(bob).toBeNull() // Deleted
      expect(charlie).not.toBeNull()
      expect(embeddingsEqual(charlie!.embedding, embedding3)).toBe(true)

      await engine2.close()
    })

    it('multiple reopens maintain consistency', async () => {
      const paths = createTestPaths('crash-multi-reopen')
      testPathsList.push(paths)

      // Session 1: Insert
      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      const embedding1 = generateRandomEmbedding(384)
      await engine1.writeRecord('counter', embedding1)
      await engine1.close()

      // Session 2: Update
      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      const embedding2 = generateRandomEmbedding(384)
      await engine2.writeRecord('counter', embedding2)
      await engine2.close()

      // Session 3: Update again
      const engine3 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      const embedding3 = generateRandomEmbedding(384)
      await engine3.writeRecord('counter', embedding3)
      await engine3.close()

      // Session 4: Verify
      const engine4 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const record = await engine4.readRecord('counter')
      expect(record).not.toBeNull()
      expect(embeddingsEqual(record!.embedding, embedding3)).toBe(true)
      expect(engine4.count()).toBe(1)

      await engine4.close()
    })
  })
})
