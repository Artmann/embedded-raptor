import { describe, it, expect, beforeEach } from 'vitest'

import { LRUCache } from './lru-cache'

describe('LRUCache', () => {
  let cache: LRUCache<string, number>

  beforeEach(() => {
    cache = new LRUCache(3)
  })

  describe('constructor', () => {
    it('should create a cache with specified maxSize', () => {
      const customCache = new LRUCache<string, number>(10)

      expect(customCache.getMaxSize()).toBe(10)
      expect(customCache.size()).toBe(0)
    })

    it('should throw error for zero maxSize', () => {
      expect(() => new LRUCache(0)).toThrow(
        'maxSize must be a positive integer'
      )
    })

    it('should throw error for negative maxSize', () => {
      expect(() => new LRUCache(-1)).toThrow(
        'maxSize must be a positive integer'
      )
    })

    it('should throw error for non-integer maxSize', () => {
      expect(() => new LRUCache(1.5)).toThrow('maxSize must be an integer')
    })
  })

  describe('set and get', () => {
    it('should store and retrieve values', () => {
      cache.set('a', 1)
      cache.set('b', 2)
      cache.set('c', 3)

      expect(cache.get('a')).toBe(1)
      expect(cache.get('b')).toBe(2)
      expect(cache.get('c')).toBe(3)
    })

    it('should return undefined for missing keys', () => {
      cache.set('a', 1)

      expect(cache.get('nonexistent')).toBeUndefined()
    })

    it('should update existing keys', () => {
      cache.set('a', 1)
      cache.set('a', 10)

      expect(cache.get('a')).toBe(10)
      expect(cache.size()).toBe(1)
    })
  })

  describe('LRU eviction', () => {
    it('should evict LRU entry when at capacity', () => {
      cache.set('a', 1)
      cache.set('b', 2)
      cache.set('c', 3)

      expect(cache.size()).toBe(3)

      // Add new entry - should evict 'a' (LRU)
      cache.set('d', 4)

      expect(cache.size()).toBe(3)
      expect(cache.get('a')).toBeUndefined()
      expect(cache.get('b')).toBe(2)
      expect(cache.get('c')).toBe(3)
      expect(cache.get('d')).toBe(4)
    })

    it('should move accessed entries to MRU position on get', () => {
      cache.set('a', 1)
      cache.set('b', 2)
      cache.set('c', 3)

      // Access 'a' - moves it to MRU position
      cache.get('a')

      // Add new entry - should evict 'b' (now LRU)
      cache.set('d', 4)

      expect(cache.get('a')).toBe(1)
      expect(cache.get('b')).toBeUndefined()
      expect(cache.get('c')).toBe(3)
      expect(cache.get('d')).toBe(4)
    })

    it('should move updated entries to MRU position on set', () => {
      cache.set('a', 1)
      cache.set('b', 2)
      cache.set('c', 3)

      // Update 'a' - moves it to MRU position
      cache.set('a', 10)

      // Add new entry - should evict 'b' (now LRU)
      cache.set('d', 4)

      expect(cache.get('a')).toBe(10)
      expect(cache.get('b')).toBeUndefined()
      expect(cache.get('c')).toBe(3)
      expect(cache.get('d')).toBe(4)
    })
  })

  describe('has', () => {
    it('should return true for existing keys', () => {
      cache.set('a', 1)

      expect(cache.has('a')).toBe(true)
    })

    it('should return false for missing keys', () => {
      expect(cache.has('nonexistent')).toBe(false)
    })

    it('should not affect LRU order', () => {
      cache.set('a', 1)
      cache.set('b', 2)
      cache.set('c', 3)

      // Check 'a' - should NOT move it to MRU
      cache.has('a')

      // Add new entry - should still evict 'a' (still LRU)
      cache.set('d', 4)

      expect(cache.get('a')).toBeUndefined()
      expect(cache.get('b')).toBe(2)
      expect(cache.get('c')).toBe(3)
      expect(cache.get('d')).toBe(4)
    })
  })

  describe('size and getMaxSize', () => {
    it('should track size correctly', () => {
      expect(cache.size()).toBe(0)

      cache.set('a', 1)

      expect(cache.size()).toBe(1)

      cache.set('b', 2)
      cache.set('c', 3)

      expect(cache.size()).toBe(3)

      cache.set('d', 4)

      expect(cache.size()).toBe(3) // Still 3 due to eviction
    })

    it('should return correct maxSize', () => {
      expect(cache.getMaxSize()).toBe(3)
    })
  })

  describe('clear', () => {
    it('should remove all entries', () => {
      cache.set('a', 1)
      cache.set('b', 2)
      cache.set('c', 3)

      expect(cache.size()).toBe(3)

      cache.clear()

      expect(cache.size()).toBe(0)
      expect(cache.get('a')).toBeUndefined()
      expect(cache.get('b')).toBeUndefined()
      expect(cache.get('c')).toBeUndefined()
    })
  })

  describe('with Float32Array values', () => {
    it('should work with Float32Array values', () => {
      const floatCache = new LRUCache<string, Float32Array>(2)
      const embedding1 = new Float32Array([0.1, 0.2, 0.3])
      const embedding2 = new Float32Array([0.4, 0.5, 0.6])
      const embedding3 = new Float32Array([0.7, 0.8, 0.9])

      floatCache.set('text1', embedding1)
      floatCache.set('text2', embedding2)

      expect(floatCache.get('text1')).toEqual(embedding1)
      expect(floatCache.get('text2')).toEqual(embedding2)

      // Add third entry - should evict text1
      floatCache.set('text3', embedding3)

      expect(floatCache.get('text1')).toBeUndefined()
      expect(floatCache.get('text2')).toEqual(embedding2)
      expect(floatCache.get('text3')).toEqual(embedding3)
    })
  })

  describe('edge cases', () => {
    it('should handle cache size of 1', () => {
      const singleCache = new LRUCache<string, number>(1)

      singleCache.set('a', 1)

      expect(singleCache.get('a')).toBe(1)

      singleCache.set('b', 2)

      expect(singleCache.get('a')).toBeUndefined()
      expect(singleCache.get('b')).toBe(2)
    })

    it('should handle large cache', () => {
      const largeCache = new LRUCache<number, number>(1000)

      for (let i = 0; i < 500; i++) {
        largeCache.set(i, i * 2)
      }

      expect(largeCache.size()).toBe(500)

      for (let i = 0; i < 500; i++) {
        expect(largeCache.get(i)).toBe(i * 2)
      }
    })

    it('should handle duplicate set calls without growing', () => {
      cache.set('a', 1)
      cache.set('a', 2)
      cache.set('a', 3)

      expect(cache.size()).toBe(1)
      expect(cache.get('a')).toBe(3)
    })

    it('should handle get on non-existent key without side effects', () => {
      cache.set('a', 1)

      expect(cache.get('nonexistent')).toBeUndefined()
      expect(cache.size()).toBe(1)
    })
  })
})
