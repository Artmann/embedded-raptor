/* eslint-disable @typescript-eslint/no-non-null-assertion */
/**
 * Concurrent read tests for StorageEngine.
 * Tests that read operations work without acquiring locks.
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

describe('StorageEngine concurrent reads', () => {
  const testPathsList: TestPaths[] = []
  const engines: StorageEngine[] = []

  afterEach(async () => {
    // Close all engines
    for (const engine of engines) {
      try {
        await engine.close()
      } catch {
        // Ignore errors during cleanup
      }
    }
    engines.length = 0

    await cleanup(testPathsList)
    testPathsList.length = 0
  })

  it('opens existing database and reads successfully', async () => {
    const paths = createTestPaths('concurrent-open')
    testPathsList.push(paths)
    const embedding = generateRandomEmbedding(384)

    // Create database with writer
    const writer = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await writer.writeRecord('test-key', embedding)
    await writer.close()

    // Open another engine and read
    const reader = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(reader)

    expect(reader.hasWriteLock()).toBe(false)
    expect(reader.count()).toBe(1)
    expect(reader.hasKey('test-key')).toBe(true)

    const record = await reader.readRecord('test-key')
    expect(record).not.toBeNull()
    expect(embeddingsEqual(record!.embedding, embedding)).toBe(true)
  })

  it('allows multiple engines to read same database concurrently without locking', async () => {
    const paths = createTestPaths('concurrent-multi-read')
    testPathsList.push(paths)
    const embedding = generateRandomEmbedding(384)

    // Create database with writer
    const writer = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await writer.writeRecord('shared-key', embedding)
    await writer.close()

    // Open multiple instances concurrently - no lock needed for reads
    const reader1 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(reader1)

    const reader2 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(reader2)

    const reader3 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(reader3)

    // None should have write lock (no writes performed)
    expect(reader1.hasWriteLock()).toBe(false)
    expect(reader2.hasWriteLock()).toBe(false)
    expect(reader3.hasWriteLock()).toBe(false)

    // All readers should see the same data
    expect(reader1.count()).toBe(1)
    expect(reader2.count()).toBe(1)
    expect(reader3.count()).toBe(1)

    const record1 = await reader1.readRecord('shared-key')
    const record2 = await reader2.readRecord('shared-key')
    const record3 = await reader3.readRecord('shared-key')

    expect(record1).not.toBeNull()
    expect(record2).not.toBeNull()
    expect(record3).not.toBeNull()

    expect(embeddingsEqual(record1!.embedding, embedding)).toBe(true)
    expect(embeddingsEqual(record2!.embedding, embedding)).toBe(true)
    expect(embeddingsEqual(record3!.embedding, embedding)).toBe(true)
  })

  it('read operations do not acquire lock', async () => {
    const paths = createTestPaths('concurrent-no-lock-on-read')
    testPathsList.push(paths)
    const embedding = generateRandomEmbedding(384)

    // Create database with writer
    const writer = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await writer.writeRecord('test-key', embedding)
    await writer.close()

    // Open reader
    const reader = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(reader)

    // Perform all read operations
    expect(reader.hasWriteLock()).toBe(false)

    reader.count()
    expect(reader.hasWriteLock()).toBe(false)

    reader.hasKey('test-key')
    expect(reader.hasWriteLock()).toBe(false)

    await reader.readRecord('test-key')
    expect(reader.hasWriteLock()).toBe(false)

    Array.from(reader.keys())
    expect(reader.hasWriteLock()).toBe(false)

    Array.from(reader.locations())
    expect(reader.hasWriteLock()).toBe(false)
  })

  it('readEmbeddingAt works correctly', async () => {
    const paths = createTestPaths('concurrent-read-embedding')
    testPathsList.push(paths)
    const embedding = generateRandomEmbedding(384)

    // Create database with writer
    const writer = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await writer.writeRecord('test-key', embedding)
    await writer.close()

    // Open reader
    const reader = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(reader)

    // Get location and read embedding directly
    const locations = Array.from(reader.locations())
    expect(locations.length).toBe(1)

    const [key, location] = locations[0]
    expect(key).toBe('test-key')

    const readEmbedding = await reader.readEmbeddingAt(location.offset)
    expect(readEmbedding).not.toBeNull()
    expect(embeddingsEqual(readEmbedding!, embedding)).toBe(true)

    // Still no lock acquired
    expect(reader.hasWriteLock()).toBe(false)
  })

  it('close is safe on engine that only performed reads', async () => {
    const paths = createTestPaths('concurrent-close')
    testPathsList.push(paths)

    // Create database
    const writer = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await writer.writeRecord('test', generateRandomEmbedding(384))
    await writer.close()

    // Open reader
    const reader = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    // Do some reads
    await reader.readRecord('test')

    // Close should not throw
    await reader.close()

    // Double close should also be safe
    await reader.close()
  })

  it('same engine can do both reads and writes', async () => {
    const paths = createTestPaths('concurrent-read-write-same-engine')
    testPathsList.push(paths)
    const embedding1 = generateRandomEmbedding(384)
    const embedding2 = generateRandomEmbedding(384)

    // Create engine
    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(engine)

    // No lock yet
    expect(engine.hasWriteLock()).toBe(false)

    // Write something - this acquires lock
    await engine.writeRecord('key1', embedding1)
    expect(engine.hasWriteLock()).toBe(true)

    // Read it back
    const record1 = await engine.readRecord('key1')
    expect(record1).not.toBeNull()
    expect(embeddingsEqual(record1!.embedding, embedding1)).toBe(true)

    // Write more
    await engine.writeRecord('key2', embedding2)

    // Read all
    expect(engine.count()).toBe(2)
    const record2 = await engine.readRecord('key2')
    expect(record2).not.toBeNull()
    expect(embeddingsEqual(record2!.embedding, embedding2)).toBe(true)
  })
})
