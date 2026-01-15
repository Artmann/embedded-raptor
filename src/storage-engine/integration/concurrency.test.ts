/* eslint-disable @typescript-eslint/no-non-null-assertion */
/**
 * Concurrency integration tests for StorageEngine.
 * Tests concurrent operations and thread safety.
 */

import { describe, it, expect, afterEach } from 'vitest'
import { StorageEngine } from '../storage-engine'
import {
  createTestPaths,
  cleanup,
  generateRandomEmbedding,
  embeddingsEqual,
  type TestPaths
} from './helpers'

describe('StorageEngine concurrency', () => {
  const testPathsList: TestPaths[] = []

  afterEach(async () => {
    await cleanup(testPathsList)
    testPathsList.length = 0
  })

  it('concurrent inserts all succeed with unique keys', async () => {
    const paths = createTestPaths('concurrent-inserts')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    const count = 50
    const embeddings = Array.from({ length: count }, () =>
      generateRandomEmbedding(384)
    )

    const promises = embeddings.map((embedding, i) =>
      engine.writeRecord(`key${i}`, embedding)
    )

    await Promise.all(promises)

    // All inserts should succeed
    expect(engine.count()).toBe(count)

    // All keys should be present
    const keys = Array.from(engine.keys())
    expect(keys).toHaveLength(count)

    // Verify all embeddings are stored correctly
    for (let i = 0; i < count; i++) {
      const record = await engine.readRecord(`key${i}`)
      expect(record).not.toBeNull()
      expect(embeddingsEqual(record!.embedding, embeddings[i])).toBe(true)
    }

    await engine.close()
  })

  it('concurrent updates to different keys succeed', async () => {
    const paths = createTestPaths('concurrent-updates')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    // Insert records sequentially first
    const keyCount = 20
    for (let i = 0; i < keyCount; i++) {
      await engine.writeRecord(`key${i}`, generateRandomEmbedding(384))
    }

    // Now update each key concurrently
    const updatedEmbeddings = Array.from({ length: keyCount }, () =>
      generateRandomEmbedding(384)
    )

    const updatePromises = updatedEmbeddings.map((embedding, i) =>
      engine.writeRecord(`key${i}`, embedding)
    )

    await Promise.all(updatePromises)

    // Verify all keys have updated embeddings
    for (let i = 0; i < keyCount; i++) {
      const record = await engine.readRecord(`key${i}`)
      expect(record).not.toBeNull()
      expect(embeddingsEqual(record!.embedding, updatedEmbeddings[i])).toBe(
        true
      )
    }

    await engine.close()
  })

  it('concurrent mixed operations maintain consistency', async () => {
    const paths = createTestPaths('concurrent-mixed')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    // Insert 30 embeddings
    const operations: Promise<unknown>[] = []
    for (let i = 0; i < 30; i++) {
      operations.push(
        engine.writeRecord(`key${i}`, generateRandomEmbedding(384))
      )
    }

    await Promise.all(operations)

    // Now do concurrent updates and deletes
    const mixedOps: Promise<unknown>[] = []

    // Update some (0-9)
    for (let i = 0; i < 10; i++) {
      mixedOps.push(engine.writeRecord(`key${i}`, generateRandomEmbedding(384)))
    }

    // Delete some (20-29)
    for (let i = 20; i < 30; i++) {
      mixedOps.push(engine.deleteRecord(`key${i}`))
    }

    // Insert more (30-39)
    for (let i = 30; i < 40; i++) {
      mixedOps.push(engine.writeRecord(`key${i}`, generateRandomEmbedding(384)))
    }

    await Promise.all(mixedOps)

    // Verify final state
    // Should have: 20 original (0-19) + 10 new (30-39) - 10 deleted (20-29) = 30
    expect(engine.count()).toBe(30)

    // Check updated keys exist (0-9)
    for (let i = 0; i < 10; i++) {
      expect(engine.hasKey(`key${i}`)).toBe(true)
    }

    // Check unchanged keys exist (10-19)
    for (let i = 10; i < 20; i++) {
      expect(engine.hasKey(`key${i}`)).toBe(true)
    }

    // Check deleted keys are gone (20-29)
    for (let i = 20; i < 30; i++) {
      expect(engine.hasKey(`key${i}`)).toBe(false)
    }

    // Check new keys exist (30-39)
    for (let i = 30; i < 40; i++) {
      expect(engine.hasKey(`key${i}`)).toBe(true)
    }

    await engine.close()
  })

  it('high contention stress test', async () => {
    const paths = createTestPaths('stress-test')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    const batches = 10
    const perBatch = 20

    // Launch all batches concurrently
    const batchPromises = Array.from({ length: batches }, (_, batch) => {
      const inserts = Array.from({ length: perBatch }, (_, index) =>
        engine.writeRecord(
          `batch${batch}-key${index}`,
          generateRandomEmbedding(384)
        )
      )
      return Promise.all(inserts)
    })

    await Promise.all(batchPromises)

    // Verify all inserts succeeded
    expect(engine.count()).toBe(batches * perBatch)

    // Verify each batch has all its keys
    for (let batch = 0; batch < batches; batch++) {
      for (let index = 0; index < perBatch; index++) {
        expect(engine.hasKey(`batch${batch}-key${index}`)).toBe(true)
      }
    }

    await engine.close()
  })

  it('concurrent reads dont block writes', async () => {
    const paths = createTestPaths('read-write')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    // Insert some initial data
    for (let i = 0; i < 10; i++) {
      await engine.writeRecord(`key${i}`, generateRandomEmbedding(384))
    }

    // Perform concurrent reads and writes
    const operations: Promise<unknown>[] = []

    // Reads
    for (let i = 0; i < 20; i++) {
      operations.push(engine.readRecord(`key${i % 10}`))
    }

    // Writes interspersed
    for (let i = 10; i < 20; i++) {
      operations.push(
        engine.writeRecord(`key${i}`, generateRandomEmbedding(384))
      )
    }

    await Promise.all(operations)

    // Verify final count
    expect(engine.count()).toBe(20)

    await engine.close()
  })

  it('data persists correctly after concurrent operations', async () => {
    const paths = createTestPaths('persist-concurrent')
    testPathsList.push(paths)

    const embeddings: Float32Array[] = []

    // First session: concurrent inserts
    const engine1 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    const count = 50
    for (let i = 0; i < count; i++) {
      embeddings.push(generateRandomEmbedding(384))
    }

    const promises = embeddings.map((embedding, i) =>
      engine1.writeRecord(`key${i}`, embedding)
    )
    await Promise.all(promises)
    await engine1.close()

    // Second session: verify all data persisted
    const engine2 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    expect(engine2.count()).toBe(count)

    // Verify all embeddings are correct
    for (let i = 0; i < count; i++) {
      const record = await engine2.readRecord(`key${i}`)
      expect(record).not.toBeNull()
      expect(embeddingsEqual(record!.embedding, embeddings[i])).toBe(true)
    }

    await engine2.close()
  })

  it('sequence numbers remain unique under concurrent writes', async () => {
    const paths = createTestPaths('seq-concurrent')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    // Concurrent writes
    const count = 100
    const promises = Array.from({ length: count }, (_, i) =>
      engine.writeRecord(`key${i}`, generateRandomEmbedding(384))
    )
    await Promise.all(promises)

    // Verify all sequence numbers are unique
    const seqNums = new Set<bigint>()
    for (const [, location] of engine.locations()) {
      expect(seqNums.has(location.sequenceNumber)).toBe(false)
      seqNums.add(location.sequenceNumber)
    }

    expect(seqNums.size).toBe(count)

    await engine.close()
  })

  it('handles rapid open/close cycles', async () => {
    const paths = createTestPaths('open-close-cycle')
    testPathsList.push(paths)

    const embedding = generateRandomEmbedding(384)

    // Initial write
    const engine1 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await engine1.writeRecord('persistent', embedding)
    await engine1.close()

    // Rapid open/close cycles
    for (let i = 0; i < 10; i++) {
      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })
      expect(engine.hasKey('persistent')).toBe(true)
      await engine.close()
    }

    // Final verification
    const engineFinal = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    const record = await engineFinal.readRecord('persistent')
    expect(record).not.toBeNull()
    expect(embeddingsEqual(record!.embedding, embedding)).toBe(true)
    await engineFinal.close()
  })
})
