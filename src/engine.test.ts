import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { EmbeddingEngine } from './engine'
import { unlink, mkdir, stat, rm } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { dirname } from 'node:path'

describe('EmbeddingEngine', () => {
  const testStorePath = './test-data/test-embeddings.raptor'
  const testWalPath = './test-data/test-embeddings.raptor-wal'
  const testLockPath = './test-data/test-embeddings.raptor.lock'
  let engine: EmbeddingEngine

  beforeEach(async () => {
    // Create test directory if it doesn't exist
    const dir = dirname(testStorePath)
    if (!existsSync(dir)) {
      await mkdir(dir, { recursive: true })
    }

    engine = new EmbeddingEngine({
      storePath: testStorePath
    })
  })

  afterEach(async () => {
    // Dispose engine first to release file locks
    try {
      await engine.dispose()
    } catch {
      // Ignore errors
    }

    // Clean up test files
    try {
      if (existsSync(testStorePath)) {
        await unlink(testStorePath)
      }
      if (existsSync(testWalPath)) {
        await unlink(testWalPath)
      }
      if (existsSync(testLockPath)) {
        await rm(testLockPath, { force: true })
      }
    } catch {
      // Ignore errors if files don't exist
    }
  })

  describe('store', () => {
    it('should create data and WAL files on first store', async () => {
      await engine.store('doc1', 'Hello world')

      expect(existsSync(testStorePath)).toBe(true)
      expect(existsSync(testWalPath)).toBe(true)
    })

    it('should store entry and retrieve it back', async () => {
      await engine.store('doc1', 'Hello world')

      const entry = await engine.get('doc1')
      expect(entry).not.toBeNull()
      expect(entry?.key).toBe('doc1')
      expect(entry?.embedding).toBeInstanceOf(Array)
      expect(entry?.embedding.length).toBe(384)
    })

    it('should store multiple entries', async () => {
      await engine.store('doc1', 'The quick brown fox')
      await engine.store('doc2', 'Machine learning is powerful')
      await engine.store('doc3', 'Bun is fast')

      // Verify file size increased
      const stats = await stat(testStorePath)
      expect(stats.size).toBeGreaterThan(16) // More than just header

      // Verify we can retrieve all entries
      const entry1 = await engine.get('doc1')
      const entry2 = await engine.get('doc2')
      const entry3 = await engine.get('doc3')

      expect(entry1?.key).toBe('doc1')
      expect(entry2?.key).toBe('doc2')
      expect(entry3?.key).toBe('doc3')
    })

    it('should update existing key (via WAL)', async () => {
      await engine.store('doc1', 'Original text')
      await engine.store('doc1', 'Updated text')

      // Get should return the latest version
      const entry = await engine.get('doc1')
      expect(entry?.key).toBe('doc1')

      // Count should be 1 (not 2) due to update
      const count = await engine.count()
      expect(count).toBe(1)
    })

    it('should handle UTF-8 keys correctly', async () => {
      await engine.store('café☕', 'Coffee text')

      const entry = await engine.get('café☕')
      expect(entry?.key).toBe('café☕')
    })
  })

  describe('storeMany', () => {
    it('should store multiple entries in batch', async () => {
      const items = [
        { key: 'doc1', text: 'The quick brown fox' },
        { key: 'doc2', text: 'Machine learning is powerful' },
        { key: 'doc3', text: 'Bun is fast' }
      ]

      await engine.storeMany(items)

      // Verify files were created
      expect(existsSync(testStorePath)).toBe(true)
      expect(existsSync(testWalPath)).toBe(true)

      // Verify all entries can be retrieved
      const entry1 = await engine.get('doc1')
      const entry2 = await engine.get('doc2')
      const entry3 = await engine.get('doc3')

      expect(entry1?.key).toBe('doc1')
      expect(entry2?.key).toBe('doc2')
      expect(entry3?.key).toBe('doc3')
    })

    it('should generate embeddings for all items', async () => {
      const items = [
        { key: 'doc1', text: 'First document' },
        { key: 'doc2', text: 'Second document' }
      ]

      await engine.storeMany(items)

      const entry1 = await engine.get('doc1')
      const entry2 = await engine.get('doc2')

      expect(entry1?.embedding.length).toBe(384)
      expect(entry2?.embedding.length).toBe(384)

      // Different texts should have different embeddings
      expect(entry1?.embedding[0]).not.toBe(entry2?.embedding[0])
    })

    it('should handle single item in array', async () => {
      const items = [{ key: 'single', text: 'Single item' }]

      await engine.storeMany(items)

      const entry = await engine.get('single')
      expect(entry?.key).toBe('single')
    })

    it('should handle UTF-8 keys in batch', async () => {
      const items = [
        { key: 'café☕', text: 'Coffee' },
        { key: '日本語', text: 'Japanese text' }
      ]

      await engine.storeMany(items)

      const entry1 = await engine.get('café☕')
      const entry2 = await engine.get('日本語')

      expect(entry1?.key).toBe('café☕')
      expect(entry2?.key).toBe('日本語')
    })

    it('should work with search after storeMany', async () => {
      const items = [
        { key: 'doc1', text: 'Machine learning is powerful' },
        { key: 'doc2', text: 'Artificial intelligence applications' },
        { key: 'doc3', text: 'Cooking recipes and food' }
      ]

      await engine.storeMany(items)

      const results = await engine.search('AI and ML', 3)

      expect(results.length).toBeGreaterThan(0)
      expect(results[0].similarity).toBeGreaterThanOrEqual(0)
    })
  })

  describe('get', () => {
    it('should retrieve stored entry by key', async () => {
      await engine.store('doc1', 'Test content')

      const entry = await engine.get('doc1')

      expect(entry).not.toBeNull()
      expect(entry?.key).toBe('doc1')
      expect(entry?.embedding).toBeInstanceOf(Array)
      expect(entry?.embedding.length).toBe(384)
    })

    it('should return null for non-existent key', async () => {
      const entry = await engine.get('nonexistent')
      expect(entry).toBeNull()
    })

    it('should return most recent entry for updated keys', async () => {
      const text1 = 'First version content here'
      const text2 = 'Second version completely different text'

      await engine.store('doc1', text1)
      await engine.store('doc1', text2)

      const entry = await engine.get('doc1')

      expect(entry).not.toBeNull()
      expect(entry?.key).toBe('doc1')

      // Verify it's the second version by checking the embedding
      const secondEmbedding = await engine.generateEmbedding(text2)

      // Check that retrieved embedding matches the second one
      expect(entry?.embedding[0]).toBeCloseTo(secondEmbedding[0])
    })

    it('should handle empty database gracefully', async () => {
      // No store calls, just query
      const entry = await engine.get('anykey')
      expect(entry).toBeNull()
    })
  })

  describe('delete', () => {
    it('should delete an existing entry', async () => {
      await engine.store('doc1', 'Test content')

      expect(await engine.has('doc1')).toBe(true)

      const deleted = await engine.delete('doc1')

      expect(deleted).toBe(true)
      expect(await engine.has('doc1')).toBe(false)
      expect(await engine.get('doc1')).toBeNull()
    })

    it('should return false for non-existent key', async () => {
      const deleted = await engine.delete('nonexistent')
      expect(deleted).toBe(false)
    })

    it('should not affect other entries', async () => {
      await engine.store('doc1', 'Content 1')
      await engine.store('doc2', 'Content 2')

      await engine.delete('doc1')

      expect(await engine.has('doc2')).toBe(true)
      expect(await engine.get('doc2')).not.toBeNull()
    })
  })

  describe('has', () => {
    it('should return true for existing key', async () => {
      await engine.store('doc1', 'Test content')

      expect(await engine.has('doc1')).toBe(true)
    })

    it('should return false for non-existent key', async () => {
      expect(await engine.has('nonexistent')).toBe(false)
    })
  })

  describe('keys', () => {
    it('should return all keys', async () => {
      await engine.store('doc1', 'Content 1')
      await engine.store('doc2', 'Content 2')
      await engine.store('doc3', 'Content 3')

      const keys = await engine.keys()

      expect(keys).toContain('doc1')
      expect(keys).toContain('doc2')
      expect(keys).toContain('doc3')
      expect(keys.length).toBe(3)
    })

    it('should return empty array for empty database', async () => {
      const keys = await engine.keys()
      expect(keys).toEqual([])
    })
  })

  describe('count', () => {
    it('should return correct count', async () => {
      expect(await engine.count()).toBe(0)

      await engine.store('doc1', 'Content 1')
      expect(await engine.count()).toBe(1)

      await engine.store('doc2', 'Content 2')
      expect(await engine.count()).toBe(2)
    })

    it('should reflect deleted entries', async () => {
      await engine.store('doc1', 'Content 1')
      await engine.store('doc2', 'Content 2')

      expect(await engine.count()).toBe(2)

      await engine.delete('doc1')

      expect(await engine.count()).toBe(1)
    })
  })

  describe('search', () => {
    beforeEach(async () => {
      await engine.store('doc1', 'The quick brown fox jumps over the lazy dog')
      await engine.store(
        'doc2',
        'Machine learning is a subset of artificial intelligence'
      )
      await engine.store('doc3', 'Bun is a fast JavaScript runtime')
    })

    it('should find similar entries', async () => {
      const results = await engine.search('machine learning')

      expect(results.length).toBeGreaterThan(0)
      expect(results[0].key).toBeDefined()
      expect(results[0].similarity).toBeTypeOf('number')
      expect(results[0].similarity).toBeGreaterThanOrEqual(-1)
      expect(results[0].similarity).toBeLessThanOrEqual(1)
    })

    it('should sort results by similarity', async () => {
      const results = await engine.search('machine learning AI', 3)

      // Verify results are sorted in descending order
      for (let i = 1; i < results.length; i++) {
        expect(results[i - 1].similarity).toBeGreaterThanOrEqual(
          results[i].similarity
        )
      }
    })

    it('should respect limit parameter', async () => {
      const results = await engine.search('test query', 2)

      expect(results.length).toBeLessThanOrEqual(2)
    })

    it('should respect minSimilarity threshold', async () => {
      const results = await engine.search('test', 10, 0.7)

      for (const result of results) {
        expect(result.similarity).toBeGreaterThanOrEqual(0.7)
      }
    })

    it('should return empty array for empty database', async () => {
      // Create a new engine with no data
      const emptyEngine = new EmbeddingEngine({
        storePath: './test-data/empty-search.raptor'
      })

      try {
        const results = await emptyEngine.search('test', 5, 0.1)
        expect(results).toEqual([])
      } finally {
        await emptyEngine.dispose()
        await rm('./test-data/empty-search.raptor', { force: true })
        await rm('./test-data/empty-search.raptor-wal', { force: true })
        await rm('./test-data/empty-search.raptor.lock', { force: true })
      }
    })

    it('should not return deleted entries', async () => {
      await engine.store('toDelete', 'artificial intelligence AI ML')

      const resultsBefore = await engine.search('machine learning', 10, 0)
      const foundBefore = resultsBefore.some((r) => r.key === 'toDelete')
      expect(foundBefore).toBe(true)

      await engine.delete('toDelete')

      const resultsAfter = await engine.search('machine learning', 10, 0)
      const foundAfter = resultsAfter.some((r) => r.key === 'toDelete')
      expect(foundAfter).toBe(false)
    })
  })
})
