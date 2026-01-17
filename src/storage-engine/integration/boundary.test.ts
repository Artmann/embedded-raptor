/* eslint-disable @typescript-eslint/no-non-null-assertion */
/**
 * Boundary condition integration tests for StorageEngine.
 * Tests edge cases and limits.
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

describe('StorageEngine boundary conditions', () => {
  const testPathsList: TestPaths[] = []

  afterEach(async () => {
    await cleanup(testPathsList)
    testPathsList.length = 0
  })

  describe('large embeddings', () => {
    it('handles 768-dimension embedding (standard transformer size)', async () => {
      const paths = createTestPaths('large-768')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 768
      })

      const embedding = generateRandomEmbedding(768)
      await engine.writeRecord('large768', embedding)

      const record = await engine.readRecord('large768')
      expect(record).not.toBeNull()
      expect(record!.embedding.length).toBe(768)
      expect(embeddingsEqual(record!.embedding, embedding)).toBe(true)

      await engine.close()
    })

    it('handles 1536-dimension embedding (OpenAI ada-002 size)', async () => {
      const paths = createTestPaths('large-1536')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 1536
      })

      const embedding = generateRandomEmbedding(1536)
      await engine.writeRecord('large1536', embedding)

      const record = await engine.readRecord('large1536')
      expect(record).not.toBeNull()
      expect(record!.embedding.length).toBe(1536)
      expect(embeddingsEqual(record!.embedding, embedding)).toBe(true)

      await engine.close()
    })

    it('handles 4096-dimension embedding', async () => {
      const paths = createTestPaths('large-4096')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 4096
      })

      const embedding = generateRandomEmbedding(4096)
      await engine.writeRecord('large4096', embedding)

      const record = await engine.readRecord('large4096')
      expect(record).not.toBeNull()
      expect(record!.embedding.length).toBe(4096)
      expect(embeddingsEqual(record!.embedding, embedding)).toBe(true)

      await engine.close()
    })

    it('large embedding persists across reopen', async () => {
      const paths = createTestPaths('large-persist')
      testPathsList.push(paths)

      const embedding = generateRandomEmbedding(1536)

      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 1536
      })
      await engine1.writeRecord('persist', embedding)
      await engine1.close()

      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 1536
      })
      const record = await engine2.readRecord('persist')

      expect(record).not.toBeNull()
      expect(embeddingsEqual(record!.embedding, embedding)).toBe(true)

      await engine2.close()
    })

    it('handles multiple large embeddings', async () => {
      const paths = createTestPaths('large-multiple')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 1536
      })

      const embeddings: Float32Array[] = []
      for (let i = 0; i < 5; i++) {
        const embedding = generateRandomEmbedding(1536)
        embeddings.push(embedding)
        await engine.writeRecord(`large${i}`, embedding)
      }

      // Verify all embeddings
      for (let i = 0; i < 5; i++) {
        const record = await engine.readRecord(`large${i}`)
        expect(record).not.toBeNull()
        expect(embeddingsEqual(record!.embedding, embeddings[i])).toBe(true)
      }

      await engine.close()
    })
  })

  describe('many embeddings', () => {
    it('handles 200 embeddings', async () => {
      const paths = createTestPaths('many-200')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      for (let i = 0; i < 200; i++) {
        await engine.writeRecord(`key${i}`, generateRandomEmbedding(384))
      }

      expect(engine.count()).toBe(200)

      // Verify random embedding
      expect(engine.hasKey('key100')).toBe(true)
      const record = await engine.readRecord('key100')
      expect(record).not.toBeNull()
      expect(record!.embedding.length).toBe(384)

      await engine.close()
    })

    it('200 embeddings persist across reopen', async () => {
      const paths = createTestPaths('many-persist')
      testPathsList.push(paths)

      const engine1 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      for (let i = 0; i < 200; i++) {
        await engine1.writeRecord(`key${i}`, generateRandomEmbedding(384))
      }
      await engine1.close()

      const engine2 = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      expect(engine2.count()).toBe(200)

      await engine2.close()
    })

    it('handles 500 embeddings with mixed operations', async () => {
      const paths = createTestPaths('many-500')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      // Insert 500
      for (let i = 0; i < 500; i++) {
        await engine.writeRecord(`key${i}`, generateRandomEmbedding(384))
      }

      // Update every 10th (50 updates)
      for (let i = 0; i < 500; i += 10) {
        await engine.writeRecord(`key${i}`, generateRandomEmbedding(384))
      }

      // Delete every 100th (5 deletes)
      for (let i = 0; i < 500; i += 100) {
        await engine.deleteRecord(`key${i}`)
      }

      // Should have: 500 - 5 deleted = 495
      expect(engine.count()).toBe(495)

      await engine.close()
    })
  })

  describe('special characters in keys', () => {
    it('handles unicode in keys', async () => {
      const paths = createTestPaths('unicode-keys')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const keys = [
        'emoji_\u{1F98D}\u{1F680}', // Gorilla, rocket
        '\u4E2D\u6587_key', // Chinese characters
        '\u0645\u0631\u062D\u0628\u0627', // Arabic
        '\u{1F1FA}\u{1F1F8}_flag' // US flag emoji
      ]

      for (const key of keys) {
        await engine.writeRecord(key, generateRandomEmbedding(384))
      }

      // Verify all keys
      for (const key of keys) {
        expect(engine.hasKey(key)).toBe(true)
        const record = await engine.readRecord(key)
        expect(record).not.toBeNull()
        expect(record!.key).toBe(key)
      }

      await engine.close()
    })

    it('handles special characters in keys', async () => {
      const paths = createTestPaths('special-keys')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const keys = [
        'key-with-dashes',
        'key_with_underscores',
        'key.with.dots',
        'key/with/slashes',
        'key:with:colons',
        'key@with@at',
        'key#with#hash'
      ]

      for (const key of keys) {
        await engine.writeRecord(key, generateRandomEmbedding(384))
      }

      // Verify all keys
      for (const key of keys) {
        expect(engine.hasKey(key)).toBe(true)
      }

      await engine.close()
    })

    it('handles long key names', async () => {
      const paths = createTestPaths('long-keys')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const longKey = 'key_' + 'x'.repeat(1000)
      const embedding = generateRandomEmbedding(384)
      await engine.writeRecord(longKey, embedding)

      const record = await engine.readRecord(longKey)
      expect(record).not.toBeNull()
      expect(record!.key).toBe(longKey)
      expect(embeddingsEqual(record!.embedding, embedding)).toBe(true)

      await engine.close()
    })

    it('key names are case-sensitive', async () => {
      const paths = createTestPaths('case-sensitive')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const embedding1 = generateRandomEmbedding(384)
      const embedding2 = generateRandomEmbedding(384)
      const embedding3 = generateRandomEmbedding(384)

      await engine.writeRecord('document', embedding1)
      await engine.writeRecord('Document', embedding2)
      await engine.writeRecord('DOCUMENT', embedding3)

      expect(engine.count()).toBe(3)

      const record1 = await engine.readRecord('document')
      const record2 = await engine.readRecord('Document')
      const record3 = await engine.readRecord('DOCUMENT')

      expect(record1).not.toBeNull()
      expect(record2).not.toBeNull()
      expect(record3).not.toBeNull()

      expect(embeddingsEqual(record1!.embedding, embedding1)).toBe(true)
      expect(embeddingsEqual(record2!.embedding, embedding2)).toBe(true)
      expect(embeddingsEqual(record3!.embedding, embedding3)).toBe(true)

      await engine.close()
    })
  })

  describe('embedding value edge cases', () => {
    it('handles embeddings with extreme values', async () => {
      const paths = createTestPaths('extreme-values')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 10
      })

      // Create embedding with extreme float values
      const embedding = new Float32Array([
        3.4028235e38, // Max float32
        -3.4028235e38, // Min float32
        1.17549435e-38, // Smallest positive normal
        0,
        -0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.1920929e-7 // Smallest positive subnormal
      ])

      await engine.writeRecord('extreme', embedding)

      const record = await engine.readRecord('extreme')
      expect(record).not.toBeNull()
      expect(record!.embedding.length).toBe(10)

      // Float32 values should match (accounting for precision)
      for (let i = 0; i < 10; i++) {
        expect(record!.embedding[i]).toBeCloseTo(embedding[i], 6)
      }

      await engine.close()
    })

    it('handles embeddings with all zeros', async () => {
      const paths = createTestPaths('zero-embedding')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const embedding = new Float32Array(384).fill(0)
      await engine.writeRecord('zeros', embedding)

      const record = await engine.readRecord('zeros')
      expect(record).not.toBeNull()
      expect(record!.embedding.every((v) => v === 0)).toBe(true)

      await engine.close()
    })

    it('handles embeddings with all ones', async () => {
      const paths = createTestPaths('ones-embedding')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const embedding = new Float32Array(384).fill(1)
      await engine.writeRecord('ones', embedding)

      const record = await engine.readRecord('ones')
      expect(record).not.toBeNull()
      expect(record!.embedding.every((v) => v === 1)).toBe(true)

      await engine.close()
    })

    it('handles embeddings with NaN values correctly', async () => {
      const paths = createTestPaths('nan-embedding')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 5
      })

      // Note: NaN handling depends on serialization
      const embedding = new Float32Array([1, NaN, 3, NaN, 5])
      await engine.writeRecord('nan', embedding)

      const record = await engine.readRecord('nan')
      expect(record).not.toBeNull()
      expect(record!.embedding[0]).toBe(1)
      expect(Number.isNaN(record!.embedding[1])).toBe(true)
      expect(record!.embedding[2]).toBe(3)
      expect(Number.isNaN(record!.embedding[3])).toBe(true)
      expect(record!.embedding[4]).toBe(5)

      await engine.close()
    })

    it('handles embeddings with Infinity values', async () => {
      const paths = createTestPaths('infinity-embedding')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 4
      })

      const embedding = new Float32Array([Infinity, -Infinity, 1.0, -1.0])
      await engine.writeRecord('infinity', embedding)

      const record = await engine.readRecord('infinity')
      expect(record).not.toBeNull()
      expect(record!.embedding[0]).toBe(Infinity)
      expect(record!.embedding[1]).toBe(-Infinity)
      expect(record!.embedding[2]).toBe(1.0)
      expect(record!.embedding[3]).toBe(-1.0)

      await engine.close()
    })
  })

  describe('dimension edge cases', () => {
    it('handles dimension 1', async () => {
      const paths = createTestPaths('dim-1')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 1
      })

      const embedding = new Float32Array([0.5])
      await engine.writeRecord('single', embedding)

      const record = await engine.readRecord('single')
      expect(record).not.toBeNull()
      expect(record!.embedding.length).toBe(1)
      expect(record!.embedding[0]).toBe(0.5)

      await engine.close()
    })

    it('handles small dimension (8)', async () => {
      const paths = createTestPaths('dim-8')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 8
      })

      const embedding = generateRandomEmbedding(8)
      await engine.writeRecord('small', embedding)

      const record = await engine.readRecord('small')
      expect(record).not.toBeNull()
      expect(embeddingsEqual(record!.embedding, embedding)).toBe(true)

      await engine.close()
    })
  })

  describe('empty key handling', () => {
    it('handles empty string key', async () => {
      const paths = createTestPaths('empty-key')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const embedding = generateRandomEmbedding(384)
      await engine.writeRecord('', embedding)

      expect(engine.hasKey('')).toBe(true)
      const record = await engine.readRecord('')
      expect(record).not.toBeNull()
      expect(record!.key).toBe('')
      expect(embeddingsEqual(record!.embedding, embedding)).toBe(true)

      await engine.close()
    })

    it('distinguishes empty key from other keys', async () => {
      const paths = createTestPaths('empty-vs-other')
      testPathsList.push(paths)

      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const emptyEmbedding = generateRandomEmbedding(384)
      const otherEmbedding = generateRandomEmbedding(384)

      await engine.writeRecord('', emptyEmbedding)
      await engine.writeRecord('other', otherEmbedding)

      expect(engine.count()).toBe(2)

      const emptyRecord = await engine.readRecord('')
      const otherRecord = await engine.readRecord('other')

      expect(embeddingsEqual(emptyRecord!.embedding, emptyEmbedding)).toBe(true)
      expect(embeddingsEqual(otherRecord!.embedding, otherEmbedding)).toBe(true)

      await engine.close()
    })
  })
})
