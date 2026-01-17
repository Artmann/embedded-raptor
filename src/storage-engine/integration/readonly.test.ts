/* eslint-disable @typescript-eslint/no-non-null-assertion */
/**
 * Read-only mode integration tests for StorageEngine.
 * Tests that read-only mode allows concurrent reads without locks.
 */

import { describe, it, expect, afterEach } from 'vitest'
import { StorageEngine } from '../storage-engine'
import { ReadOnlyError } from '../file-lock'
import {
  createTestPaths,
  cleanup,
  generateRandomEmbedding,
  embeddingsEqual,
  type TestPaths
} from './helpers'

describe('StorageEngine read-only mode', () => {
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

  it('opens existing database in read-only mode successfully', async () => {
    const paths = createTestPaths('readonly-open')
    testPathsList.push(paths)
    const embedding = generateRandomEmbedding(384)

    // Create database with writer
    const writer = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await writer.writeRecord('test-key', embedding)
    await writer.close()

    // Open in read-only mode
    const reader = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384,
      readOnly: true
    })
    engines.push(reader)

    expect(reader.isReadOnly()).toBe(true)
    expect(reader.count()).toBe(1)
    expect(reader.hasKey('test-key')).toBe(true)

    const record = await reader.readRecord('test-key')
    expect(record).not.toBeNull()
    expect(embeddingsEqual(record!.embedding, embedding)).toBe(true)
  })

  it('throws ReadOnlyError on writeRecord in read-only mode', async () => {
    const paths = createTestPaths('readonly-write-error')
    testPathsList.push(paths)

    // Create empty database
    const writer = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await writer.writeRecord('initial', generateRandomEmbedding(384))
    await writer.close()

    // Open in read-only mode
    const reader = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384,
      readOnly: true
    })
    engines.push(reader)

    // Attempt to write should throw ReadOnlyError
    await expect(
      reader.writeRecord('new-key', generateRandomEmbedding(384))
    ).rejects.toThrow(ReadOnlyError)
  })

  it('throws ReadOnlyError on deleteRecord in read-only mode', async () => {
    const paths = createTestPaths('readonly-delete-error')
    testPathsList.push(paths)

    // Create database with a record
    const writer = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await writer.writeRecord('to-delete', generateRandomEmbedding(384))
    await writer.close()

    // Open in read-only mode
    const reader = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384,
      readOnly: true
    })
    engines.push(reader)

    // Attempt to delete should throw ReadOnlyError
    await expect(reader.deleteRecord('to-delete')).rejects.toThrow(
      ReadOnlyError
    )
  })

  it('allows multiple read-only engines to open same database concurrently', async () => {
    const paths = createTestPaths('readonly-concurrent')
    testPathsList.push(paths)
    const embedding = generateRandomEmbedding(384)

    // Create database with writer
    const writer = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await writer.writeRecord('shared-key', embedding)
    await writer.close()

    // Open multiple read-only instances concurrently
    const reader1 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384,
      readOnly: true
    })
    engines.push(reader1)

    const reader2 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384,
      readOnly: true
    })
    engines.push(reader2)

    const reader3 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384,
      readOnly: true
    })
    engines.push(reader3)

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

  it('read-only engine can coexist with active writer', async () => {
    const paths = createTestPaths('readonly-coexist-writer')
    testPathsList.push(paths)
    const embedding1 = generateRandomEmbedding(384)

    // Create database and keep writer open
    const writer = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(writer)
    await writer.writeRecord('key1', embedding1)

    // Open read-only instance while writer is still open
    const reader = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384,
      readOnly: true
    })
    engines.push(reader)

    // Reader sees data that existed when it opened
    expect(reader.count()).toBe(1)
    expect(reader.hasKey('key1')).toBe(true)

    const record = await reader.readRecord('key1')
    expect(record).not.toBeNull()
    expect(embeddingsEqual(record!.embedding, embedding1)).toBe(true)

    // Writer can continue to write
    const embedding2 = generateRandomEmbedding(384)
    await writer.writeRecord('key2', embedding2)

    // Writer sees both keys
    expect(writer.count()).toBe(2)

    // Reader still sees snapshot from when it opened (snapshot isolation)
    // Note: depending on implementation, reader may or may not see new data
    // The important thing is that it doesn't crash or corrupt data
    expect(reader.hasKey('key1')).toBe(true)
  })

  it('throws error when opening non-existent database in read-only mode', async () => {
    const paths = createTestPaths('readonly-nonexistent')
    testPathsList.push(paths)

    // Attempt to open non-existent database in read-only mode should fail
    await expect(
      StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384,
        readOnly: true
      })
    ).rejects.toThrow()
  })

  it('isReadOnly returns false for normal engines', async () => {
    const paths = createTestPaths('readonly-check-normal')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(engine)

    expect(engine.isReadOnly()).toBe(false)
  })

  it('read-only engine readEmbeddingAt works correctly', async () => {
    const paths = createTestPaths('readonly-read-embedding')
    testPathsList.push(paths)
    const embedding = generateRandomEmbedding(384)

    // Create database with writer
    const writer = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await writer.writeRecord('test-key', embedding)
    await writer.close()

    // Open in read-only mode
    const reader = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384,
      readOnly: true
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
  })

  it('close is safe on read-only engine', async () => {
    const paths = createTestPaths('readonly-close')
    testPathsList.push(paths)

    // Create database
    const writer = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await writer.writeRecord('test', generateRandomEmbedding(384))
    await writer.close()

    // Open in read-only mode
    const reader = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384,
      readOnly: true
    })

    // Close should not throw
    await reader.close()

    // Double close should also be safe
    await reader.close()
  })
})
