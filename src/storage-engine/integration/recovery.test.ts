/* eslint-disable @typescript-eslint/no-non-null-assertion */
/**
 * Recovery integration tests for StorageEngine.
 * Tests WAL recovery, index rebuild, and sequence number handling.
 */

import { describe, it, expect, afterEach } from 'vitest'
import { writeFile, readFile } from 'node:fs/promises'
import { StorageEngine } from '../storage-engine'
import { Wal } from '../wal'
import { KeyIndex } from '../key-index'
import { serializeDataRecord } from '../data-format'
import { opType, walEntrySize } from '../constants'
import {
  createTestPaths,
  cleanup,
  generateRandomEmbedding,
  collectEntries,
  flipByte,
  embeddingsEqual,
  type TestPaths
} from './helpers'

describe('StorageEngine recovery', () => {
  const testPathsList: TestPaths[] = []

  afterEach(async () => {
    await cleanup(testPathsList)
    testPathsList.length = 0
  })

  describe('index rebuild from WAL', () => {
    it('index is correctly rebuilt from WAL on startup', async () => {
      const paths = createTestPaths('rebuild')
      testPathsList.push(paths)

      const embeddings = {
        alice: generateRandomEmbedding(384),
        bob: generateRandomEmbedding(384),
        aliceUpdated: generateRandomEmbedding(384)
      }

      // Create database with multiple operations
      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      await engine1.writeRecord('alice', embeddings.alice)
      await engine1.writeRecord('bob', embeddings.bob)
      await engine1.writeRecord('alice', embeddings.aliceUpdated) // Update
      await engine1.writeRecord('charlie', generateRandomEmbedding(384))
      await engine1.close()

      // Rebuild index from WAL directly
      const wal = new Wal(paths.walPath)
      const { index } = await KeyIndex.buildFromWal(wal, paths.dataPath)
      await wal.close()

      // Verify index state
      expect(index.count()).toBe(3) // alice, bob, charlie
      expect(index.has('alice')).toBe(true)
      expect(index.has('bob')).toBe(true)
      expect(index.has('charlie')).toBe(true)

      // Verify via Database
      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const aliceRecord = await engine2.readRecord('alice')
      expect(aliceRecord).not.toBeNull()
      expect(
        embeddingsEqual(aliceRecord!.embedding, embeddings.aliceUpdated)
      ).toBe(true)

      await engine2.close()
    })

    it('recovery ignores orphaned data without WAL entry', async () => {
      const paths = createTestPaths('orphan-recovery')
      testPathsList.push(paths)

      const validEmbedding = generateRandomEmbedding(384)

      // Create valid data
      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine1.writeRecord('valid', validEmbedding)
      await engine1.close()

      // Append orphaned data to data file (no WAL entry)
      const orphanData = serializeDataRecord({
        opType: opType.insert,
        sequenceNumber: 999n,
        key: 'orphan',
        dimension: 384,
        embedding: generateRandomEmbedding(384)
      })

      const existingData = await readFile(paths.dataPath)
      const combined = new Uint8Array(existingData.length + orphanData.length)
      combined.set(existingData, 0)
      combined.set(orphanData, existingData.length)
      await writeFile(paths.dataPath, combined)

      // Recover
      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      // Only valid embedding should be found
      expect(engine2.count()).toBe(1)
      expect(engine2.hasKey('valid')).toBe(true)
      expect(engine2.hasKey('orphan')).toBe(false)

      await engine2.close()
    })

    it('recovery stops at first corrupted WAL entry', async () => {
      const paths = createTestPaths('stop-corrupt')
      testPathsList.push(paths)

      // Write valid entries via StorageEngine
      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine1.writeRecord('first', generateRandomEmbedding(384))
      await engine1.writeRecord('second', generateRandomEmbedding(384))
      await engine1.close()

      // Corrupt second WAL entry (flip a byte in the middle of the checksum area)
      const walBytes = await readFile(paths.walPath)
      const walArray = new Uint8Array(walBytes)
      flipByte(walArray, walEntrySize + 10)
      await writeFile(paths.walPath, walArray)

      // Recover - should only get first entry via WAL
      const wal = new Wal(paths.walPath)
      const entries = await collectEntries(wal.recover())
      await wal.close()

      expect(entries).toHaveLength(1)
    })

    it('recovery handles truncated files gracefully', async () => {
      const paths = createTestPaths('truncated')
      testPathsList.push(paths)

      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine1.writeRecord('first', generateRandomEmbedding(384))
      await engine1.writeRecord('second', generateRandomEmbedding(384))
      await engine1.writeRecord('third', generateRandomEmbedding(384))
      await engine1.close()

      // Truncate WAL after 1.5 records
      const walBytes = await readFile(paths.walPath)
      const truncated = walBytes.subarray(
        0,
        walEntrySize + Math.floor(walEntrySize / 2)
      )
      await writeFile(paths.walPath, truncated)

      // Recovery should succeed with first entry only
      const wal = new Wal(paths.walPath)
      const entries = await collectEntries(wal.recover())
      await wal.close()

      expect(entries).toHaveLength(1)
    })
  })

  describe('sequence number recovery', () => {
    it('sequence number resumes from last committed value', async () => {
      const paths = createTestPaths('seq-resume')
      testPathsList.push(paths)

      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine1.writeRecord('first', generateRandomEmbedding(384))
      await engine1.writeRecord('second', generateRandomEmbedding(384))
      await engine1.writeRecord('third', generateRandomEmbedding(384))
      await engine1.close()

      // Check WAL sequence numbers
      const wal = new Wal(paths.walPath)
      let maxSeq = 0n
      for await (const entry of wal.recover()) {
        if (entry.sequenceNumber > maxSeq) {
          maxSeq = entry.sequenceNumber
        }
      }
      await wal.close()

      expect(maxSeq).toBe(3n)

      // New engine should continue from 4n (next after max)
      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine2.writeRecord('fourth', generateRandomEmbedding(384))
      await engine2.close()

      const wal2 = new Wal(paths.walPath)
      let newMaxSeq = 0n
      for await (const entry of wal2.recover()) {
        if (entry.sequenceNumber > newMaxSeq) {
          newMaxSeq = entry.sequenceNumber
        }
      }
      await wal2.close()

      expect(newMaxSeq).toBe(4n)
    })

    it('sequence numbers are monotonically increasing', async () => {
      const paths = createTestPaths('seq-monotonic')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      // Multiple operations
      await engine.writeRecord('a', generateRandomEmbedding(384))
      await engine.writeRecord('b', generateRandomEmbedding(384))
      await engine.writeRecord('a', generateRandomEmbedding(384)) // Update
      await engine.writeRecord('c', generateRandomEmbedding(384))
      await engine.close()

      const wal = new Wal(paths.walPath)
      const entries = await collectEntries(wal.recover())
      await wal.close()

      // Verify monotonic increase
      for (let i = 1; i < entries.length; i++) {
        expect(entries[i].sequenceNumber).toBeGreaterThan(
          entries[i - 1].sequenceNumber
        )
      }
    })

    it('updates and deletes get new sequence numbers', async () => {
      const paths = createTestPaths('seq-updates')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      await engine.writeRecord('doc', generateRandomEmbedding(384))
      await engine.writeRecord('doc', generateRandomEmbedding(384)) // Update
      await engine.writeRecord('doc', generateRandomEmbedding(384)) // Update again
      await engine.deleteRecord('doc')
      await engine.close()

      const wal = new Wal(paths.walPath)
      const entries = await collectEntries(wal.recover())
      await wal.close()

      expect(entries).toHaveLength(4)
      expect(entries[0]?.opType).toBe(opType.insert)
      expect(entries[1]?.opType).toBe(opType.insert) // Update is also insert in our model
      expect(entries[2]?.opType).toBe(opType.insert)
      expect(entries[3]?.opType).toBe(opType.delete)

      // All sequence numbers unique and increasing
      const seqNums = entries.map((e) => e.sequenceNumber)
      expect(new Set(seqNums).size).toBe(4)
    })
  })

  describe('storage engine recovery', () => {
    it('StorageEngine.create rebuilds state correctly', async () => {
      const paths = createTestPaths('engine-create')
      testPathsList.push(paths)

      const embeddings = {
        alice: generateRandomEmbedding(384),
        bob: generateRandomEmbedding(384)
      }

      // Create data
      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      await engine1.writeRecord('alice', embeddings.alice)
      await engine1.writeRecord('bob', embeddings.bob)
      await engine1.deleteRecord('alice')
      await engine1.close()

      // Create new storage engine (simulates restart)
      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      // Verify state
      expect(engine2.hasKey('alice')).toBe(false) // Deleted
      expect(engine2.hasKey('bob')).toBe(true)
      expect(engine2.count()).toBe(1)

      const bobRecord = await engine2.readRecord('bob')
      expect(bobRecord).not.toBeNull()
      expect(embeddingsEqual(bobRecord!.embedding, embeddings.bob)).toBe(true)

      await engine2.close()
    })

    it('recovery with mixed operation types', async () => {
      const paths = createTestPaths('mixed-ops')
      testPathsList.push(paths)

      const embeddings = {
        item1: generateRandomEmbedding(384),
        item2: generateRandomEmbedding(384),
        item3: generateRandomEmbedding(384),
        item1Updated: generateRandomEmbedding(384),
        item3Updated: generateRandomEmbedding(384)
      }

      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      // Create, update, delete pattern
      await engine1.writeRecord('item1', embeddings.item1)
      await engine1.writeRecord('item2', embeddings.item2)
      await engine1.writeRecord('item3', embeddings.item3)

      await engine1.writeRecord('item1', embeddings.item1Updated) // Update
      await engine1.deleteRecord('item2')
      await engine1.writeRecord('item3', embeddings.item3Updated) // Update
      await engine1.close()

      // Recover
      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      expect(engine2.count()).toBe(2)

      const item1 = await engine2.readRecord('item1')
      const item2 = await engine2.readRecord('item2')
      const item3 = await engine2.readRecord('item3')

      expect(item1).not.toBeNull()
      expect(embeddingsEqual(item1!.embedding, embeddings.item1Updated)).toBe(
        true
      )
      expect(item2).toBeNull()
      expect(item3).not.toBeNull()
      expect(embeddingsEqual(item3!.embedding, embeddings.item3Updated)).toBe(
        true
      )

      await engine2.close()
    })
  })

  describe('empty database recovery', () => {
    it('recovers empty database with no files', async () => {
      const paths = createTestPaths('empty-no-files')
      testPathsList.push(paths)

      // Don't create any files, just create storage engine
      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      expect(engine.count()).toBe(0)
      expect(Array.from(engine.keys())).toEqual([])

      await engine.close()
    })

    it('recovers database with empty WAL', async () => {
      const paths = createTestPaths('empty-wal')
      testPathsList.push(paths)

      // Create empty files
      await writeFile(paths.dataPath, new Uint8Array(0))
      await writeFile(paths.walPath, new Uint8Array(0))

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      expect(engine.count()).toBe(0)

      // Should be able to insert new data
      const embedding = generateRandomEmbedding(384)
      await engine.writeRecord('new', embedding)

      const record = await engine.readRecord('new')
      expect(record).not.toBeNull()
      expect(embeddingsEqual(record!.embedding, embedding)).toBe(true)

      await engine.close()
    })
  })

  describe('WAL entry application order', () => {
    it('later updates override earlier inserts', async () => {
      const paths = createTestPaths('override')
      testPathsList.push(paths)

      const embeddings = {
        v1: generateRandomEmbedding(384),
        v2: generateRandomEmbedding(384),
        v3: generateRandomEmbedding(384)
      }

      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      await engine1.writeRecord('doc', embeddings.v1)
      await engine1.writeRecord('doc', embeddings.v2)
      await engine1.writeRecord('doc', embeddings.v3)
      await engine1.close()

      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const record = await engine2.readRecord('doc')
      expect(record).not.toBeNull()
      expect(embeddingsEqual(record!.embedding, embeddings.v3)).toBe(true)

      await engine2.close()
    })

    it('delete after updates results in no record', async () => {
      const paths = createTestPaths('delete-after')
      testPathsList.push(paths)

      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      await engine1.writeRecord('doc', generateRandomEmbedding(384))
      await engine1.writeRecord('doc', generateRandomEmbedding(384)) // Update
      await engine1.deleteRecord('doc')
      await engine1.close()

      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const record = await engine2.readRecord('doc')
      expect(record).toBeNull()
      expect(engine2.count()).toBe(0)

      await engine2.close()
    })
  })
})
