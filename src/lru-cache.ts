import invariant from 'tiny-invariant'

/**
 * Generic LRU (Least Recently Used) cache with O(1) get/set operations.
 * Uses JavaScript Map which maintains insertion order - LRU is first key, MRU is last.
 */
export class LRUCache<K, V> {
  private readonly cache: Map<K, V>
  private readonly maxSize: number

  /**
   * Creates a new LRU cache with the specified maximum size.
   * @param maxSize - Maximum number of entries to cache (must be positive)
   */
  constructor(maxSize: number) {
    invariant(typeof maxSize === 'number', 'maxSize must be a number')
    invariant(Number.isInteger(maxSize), 'maxSize must be an integer')
    invariant(maxSize > 0, 'maxSize must be a positive integer')

    this.maxSize = maxSize
    this.cache = new Map()
  }

  /**
   * Clears all entries from the cache.
   */
  clear(): void {
    this.cache.clear()
  }

  /**
   * Gets a value from the cache and moves it to MRU position.
   * @param key - The key to look up
   * @returns The cached value, or undefined if not found
   */
  get(key: K): V | undefined {
    const value = this.cache.get(key)

    if (value === undefined) {
      return undefined
    }

    // Move to end (MRU position) by deleting and re-inserting
    this.cache.delete(key)
    this.cache.set(key, value)

    return value
  }

  /**
   * Returns the maximum number of entries the cache can hold.
   */
  getMaxSize(): number {
    return this.maxSize
  }

  /**
   * Checks if a key exists in the cache without affecting LRU order.
   * @param key - The key to check
   * @returns true if the key exists, false otherwise
   */
  has(key: K): boolean {
    return this.cache.has(key)
  }

  /**
   * Sets a value in the cache. If the cache is at capacity, evicts the LRU entry.
   * If the key already exists, updates the value and moves to MRU position.
   * @param key - The key to set
   * @param value - The value to cache
   */
  set(key: K, value: V): void {
    if (this.cache.has(key)) {
      this.cache.delete(key)
    } else if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value as K

      this.cache.delete(firstKey)
    }

    this.cache.set(key, value)
  }

  /**
   * Returns the current number of entries in the cache.
   */
  size(): number {
    return this.cache.size
  }

  /**
   * Deletes an entry from the cache.
   * @param key - The key to delete
   * @returns true if the entry was deleted, false if it didn't exist
   */
  delete(key: K): boolean {
    return this.cache.delete(key)
  }
}
