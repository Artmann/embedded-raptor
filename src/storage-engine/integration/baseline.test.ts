/* eslint-disable @typescript-eslint/no-non-null-assertion */
/**
 * Baseline integration tests for StorageEngine.
 * Tests basic CRUD operations with real files.
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

describe('StorageEngine baseline', () => {
  const testPathsList: TestPaths[] = []

  afterEach(async () => {
    await cleanup(testPathsList)
    testPathsList.length = 0
  })

  it('should insert and read a single embedding', async () => {
    const paths = createTestPaths('baseline-insert')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    const embedding = generateRandomEmbedding(384)
    await engine.writeRecord('doc1', embedding)

    const record = await engine.readRecord('doc1')
    expect(record).not.toBeNull()
    expect(record!.key).toBe('doc1')
    expect(embeddingsEqual(record!.embedding, embedding)).toBe(true)

    await engine.close()
  })

  it('should handle multiple embeddings', async () => {
    const paths = createTestPaths('baseline-multi')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    const embeddings = {
      doc1: generateRandomEmbedding(384),
      doc2: generateRandomEmbedding(384),
      doc3: generateRandomEmbedding(384)
    }

    for (const [key, embedding] of Object.entries(embeddings)) {
      await engine.writeRecord(key, embedding)
    }

    expect(engine.count()).toBe(3)

    for (const [key, embedding] of Object.entries(embeddings)) {
      const record = await engine.readRecord(key)
      expect(record).not.toBeNull()
      expect(embeddingsEqual(record!.embedding, embedding)).toBe(true)
    }

    await engine.close()
  })

  it('should update an existing embedding', async () => {
    const paths = createTestPaths('baseline-update')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    const original = generateRandomEmbedding(384)
    const updated = generateRandomEmbedding(384)

    await engine.writeRecord('doc1', original)
    await engine.writeRecord('doc1', updated)

    const record = await engine.readRecord('doc1')
    expect(record).not.toBeNull()
    expect(embeddingsEqual(record!.embedding, updated)).toBe(true)
    expect(embeddingsEqual(record!.embedding, original)).toBe(false)

    // Count should still be 1 since it's an update
    expect(engine.count()).toBe(1)

    await engine.close()
  })

  it('should delete an embedding', async () => {
    const paths = createTestPaths('baseline-delete')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    const embedding = generateRandomEmbedding(384)
    await engine.writeRecord('doc1', embedding)

    expect(engine.hasKey('doc1')).toBe(true)
    expect(engine.count()).toBe(1)

    const deleted = await engine.deleteRecord('doc1')
    expect(deleted).toBe(true)

    expect(engine.hasKey('doc1')).toBe(false)
    expect(engine.count()).toBe(0)

    const record = await engine.readRecord('doc1')
    expect(record).toBeNull()

    await engine.close()
  })

  it('should return false when deleting non-existent key', async () => {
    const paths = createTestPaths('baseline-delete-missing')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    const deleted = await engine.deleteRecord('nonexistent')
    expect(deleted).toBe(false)

    await engine.close()
  })

  it('should persist data across reopens', async () => {
    const paths = createTestPaths('baseline-persist')
    testPathsList.push(paths)

    const embedding = generateRandomEmbedding(384)

    // First session: write data
    const engine1 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await engine1.writeRecord('doc1', embedding)
    await engine1.close()

    // Second session: read data
    const engine2 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    expect(engine2.count()).toBe(1)
    expect(engine2.hasKey('doc1')).toBe(true)

    const record = await engine2.readRecord('doc1')
    expect(record).not.toBeNull()
    expect(embeddingsEqual(record!.embedding, embedding)).toBe(true)

    await engine2.close()
  })

  it('should persist deletes across reopens', async () => {
    const paths = createTestPaths('baseline-persist-delete')
    testPathsList.push(paths)

    const embedding = generateRandomEmbedding(384)

    // First session: write and delete
    const engine1 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await engine1.writeRecord('doc1', embedding)
    await engine1.deleteRecord('doc1')
    await engine1.close()

    // Second session: verify deleted
    const engine2 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    expect(engine2.count()).toBe(0)
    expect(engine2.hasKey('doc1')).toBe(false)

    await engine2.close()
  })

  it('should iterate over all keys', async () => {
    const paths = createTestPaths('baseline-keys')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    await engine.writeRecord('alpha', generateRandomEmbedding(384))
    await engine.writeRecord('beta', generateRandomEmbedding(384))
    await engine.writeRecord('gamma', generateRandomEmbedding(384))

    const keys = Array.from(engine.keys()).sort()
    expect(keys).toEqual(['alpha', 'beta', 'gamma'])

    await engine.close()
  })

  it('should iterate over all locations', async () => {
    const paths = createTestPaths('baseline-locations')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    await engine.writeRecord('doc1', generateRandomEmbedding(384))
    await engine.writeRecord('doc2', generateRandomEmbedding(384))

    const locations = Array.from(engine.locations())
    expect(locations.length).toBe(2)

    for (const [key, location] of locations) {
      expect(typeof key).toBe('string')
      expect(typeof location.offset).toBe('number')
      expect(typeof location.length).toBe('number')
      expect(location.offset).toBeGreaterThan(0)
      expect(location.length).toBeGreaterThan(0)
    }

    await engine.close()
  })

  it('should reject embeddings with wrong dimension', async () => {
    const paths = createTestPaths('baseline-dimension')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    const wrongDimension = generateRandomEmbedding(768)

    await expect(engine.writeRecord('doc1', wrongDimension)).rejects.toThrow(
      /dimension mismatch/
    )

    await engine.close()
  })

  it('should handle empty database', async () => {
    const paths = createTestPaths('baseline-empty')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    expect(engine.count()).toBe(0)
    expect(engine.hasKey('anything')).toBe(false)
    expect(await engine.readRecord('anything')).toBeNull()
    expect(Array.from(engine.keys())).toEqual([])

    await engine.close()
  })
})
