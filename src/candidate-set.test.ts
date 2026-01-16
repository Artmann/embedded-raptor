import { describe, it, expect, beforeEach } from 'vitest'

import { CandidateSet } from './candidate-set'

describe('CandidateSet', () => {
  let candidateSet: CandidateSet

  beforeEach(() => {
    candidateSet = new CandidateSet(3)
  })

  describe('constructor', () => {
    it('should create a CandidateSet with default size 5', () => {
      const defaultSet = new CandidateSet()

      expect(defaultSet.size).toBe(5)
      expect(defaultSet.count()).toBe(0)
    })

    it('should create a CandidateSet with specified size', () => {
      const customSet = new CandidateSet(10)

      expect(customSet.size).toBe(10)
      expect(customSet.count()).toBe(0)
    })

    it('should throw error for invalid size', () => {
      expect(() => new CandidateSet(0)).toThrow(
        'Size must be a positive integer.'
      )
      expect(() => new CandidateSet(-1)).toThrow(
        'Size must be a positive integer.'
      )
    })
  })

  describe('add', () => {
    it('should add entries when under capacity', () => {
      candidateSet.add('key1', 0.8)
      candidateSet.add('key2', 0.6)
      candidateSet.add('key3', 0.9)

      expect(candidateSet.count()).toBe(3)

      const entries = candidateSet.getEntries()

      expect(entries).toHaveLength(3)
      expect(entries[0].key).toBe('key3')
      expect(entries[0].value).toBe(0.9)
      expect(entries[1].key).toBe('key1')
      expect(entries[1].value).toBe(0.8)
      expect(entries[2].key).toBe('key2')
      expect(entries[2].value).toBe(0.6)
    })

    it('should replace lowest value when at capacity', () => {
      candidateSet.add('key1', 0.5)
      candidateSet.add('key2', 0.7)
      candidateSet.add('key3', 0.3)

      expect(candidateSet.count()).toBe(3)

      // Add a higher value - should replace the lowest (0.3)
      candidateSet.add('key4', 0.6)

      expect(candidateSet.count()).toBe(3)

      const keys = candidateSet.getKeys()

      expect(keys).toContain('key2')
      expect(keys).toContain('key4')
      expect(keys).toContain('key1')
      expect(keys).not.toContain('key3')
    })

    it('should not add value lower than minimum when at capacity', () => {
      candidateSet.add('key1', 0.5)
      candidateSet.add('key2', 0.7)
      candidateSet.add('key3', 0.6)

      expect(candidateSet.count()).toBe(3)

      // Try to add a lower value - should be ignored
      candidateSet.add('key4', 0.4)

      expect(candidateSet.count()).toBe(3)

      const keys = candidateSet.getKeys()

      expect(keys).not.toContain('key4')
      expect(keys).toContain('key1')
      expect(keys).toContain('key2')
      expect(keys).toContain('key3')
    })

    it('should throw error for invalid parameters', () => {
      expect(() => candidateSet.add('', 0.5)).toThrow('Key must be provided.')
    })

    it('should accept zero as a valid value', () => {
      candidateSet.add('zero-similarity', 0)

      expect(candidateSet.count()).toBe(1)

      const entries = candidateSet.getEntries()

      expect(entries[0].key).toBe('zero-similarity')
      expect(entries[0].value).toBe(0)
    })
  })

  describe('getEntries', () => {
    it('should return entries sorted by value in descending order', () => {
      candidateSet.add('low', 0.3)
      candidateSet.add('high', 0.9)
      candidateSet.add('medium', 0.6)

      const entries = candidateSet.getEntries()

      expect(entries).toHaveLength(3)
      expect(entries[0].value).toBe(0.9)
      expect(entries[1].value).toBe(0.6)
      expect(entries[2].value).toBe(0.3)
      expect(entries[0].key).toBe('high')
      expect(entries[1].key).toBe('medium')
      expect(entries[2].key).toBe('low')
    })

    it('should return a copy of entries array', () => {
      candidateSet.add('key1', 0.5)

      const entries1 = candidateSet.getEntries()
      const entries2 = candidateSet.getEntries()

      expect(entries1).not.toBe(entries2)
      expect(entries1).toEqual(entries2)
    })

    it('should return empty array when no entries', () => {
      const entries = candidateSet.getEntries()

      expect(entries).toEqual([])
    })
  })

  describe('getKeys', () => {
    it('should return keys sorted by value in descending order', () => {
      candidateSet.add('low', 0.3)
      candidateSet.add('high', 0.9)
      candidateSet.add('medium', 0.6)

      const keys = candidateSet.getKeys()

      expect(keys).toEqual(['high', 'medium', 'low'])
    })

    it('should return empty array when no entries', () => {
      const keys = candidateSet.getKeys()

      expect(keys).toEqual([])
    })
  })

  describe('count', () => {
    it('should return correct count', () => {
      expect(candidateSet.count()).toBe(0)

      candidateSet.add('key1', 0.5)

      expect(candidateSet.count()).toBe(1)

      candidateSet.add('key2', 0.7)
      candidateSet.add('key3', 0.3)

      expect(candidateSet.count()).toBe(3)

      candidateSet.add('key4', 0.8)

      expect(candidateSet.count()).toBe(3)
    })
  })

  describe('size limits', () => {
    it('should handle size 1 correctly', () => {
      const singleSet = new CandidateSet(1)

      singleSet.add('first', 0.5)

      expect(singleSet.count()).toBe(1)
      expect(singleSet.getKeys()).toEqual(['first'])

      singleSet.add('second', 0.3)

      expect(singleSet.count()).toBe(1)
      expect(singleSet.getKeys()).toEqual(['first'])

      singleSet.add('third', 0.8)

      expect(singleSet.count()).toBe(1)
      expect(singleSet.getKeys()).toEqual(['third'])
    })

    it('should handle large size correctly', () => {
      const largeSet = new CandidateSet(100)

      for (let i = 0; i < 50; i++) {
        largeSet.add(`key${i}`, Math.random())
      }

      expect(largeSet.count()).toBe(50)

      const entries = largeSet.getEntries()

      expect(entries).toHaveLength(50)

      // Verify sorting
      for (let i = 1; i < entries.length; i++) {
        expect(entries[i - 1].value).toBeGreaterThanOrEqual(entries[i].value)
      }
    })
  })

  describe('edge cases', () => {
    it('should handle identical values', () => {
      candidateSet.add('key1', 0.5)
      candidateSet.add('key2', 0.5)
      candidateSet.add('key3', 0.5)

      expect(candidateSet.count()).toBe(3)

      candidateSet.add('key4', 0.5)

      expect(candidateSet.count()).toBe(3)

      const keys = candidateSet.getKeys()

      expect(keys).toHaveLength(3)
      expect(keys).toContain('key1')
      expect(keys).toContain('key2')
      expect(keys).toContain('key3')
    })

    it('should handle very small values', () => {
      candidateSet.add('tiny1', 0.001)
      candidateSet.add('tiny2', 0.002)
      candidateSet.add('tiny3', 0.003)

      const entries = candidateSet.getEntries()

      expect(entries[0].value).toBe(0.003)
      expect(entries[1].value).toBe(0.002)
      expect(entries[2].value).toBe(0.001)
    })

    it('should handle negative values', () => {
      candidateSet.add('negative1', -0.5)
      candidateSet.add('positive1', 0.5)
      candidateSet.add('negative2', -0.3)

      const entries = candidateSet.getEntries()

      expect(entries[0].value).toBe(0.5)
      expect(entries[1].value).toBe(-0.3)
      expect(entries[2].value).toBe(-0.5)
    })
  })
})
