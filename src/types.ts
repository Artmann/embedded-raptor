export interface EmbeddingEntry {
  key: string
  text: string
  embedding: number[]
  timestamp: number
}

export interface SearchResult {
  key: string
  similarity: number
}

export interface StoreOptions {
  storePath?: string
}

export interface EngineOptions {
  storePath: string
  /** Directory to cache downloaded models (default: ./.cache/models) */
  cacheDir?: string
  /** Open database in read-only mode (default: false). Allows concurrent reads without exclusive lock. */
  readOnly?: boolean
  /** Size of the LRU cache for text-to-embedding lookups (default: 0 = disabled) */
  embeddingCacheSize?: number
  /**
   * Enable lazy embedding loading mode (default: false).
   * When enabled, embeddings are loaded on-demand during search instead of all at once.
   * This reduces memory usage for large databases at the cost of slower first-time searches.
   */
  lazyEmbeddings?: boolean
  /**
   * Maximum number of embeddings to keep in memory when lazy mode is enabled (default: 1000).
   * Uses LRU eviction when the limit is reached. Only applies when lazyEmbeddings is true.
   */
  maxEmbeddingsInMemory?: number
}

export interface PackageJson {
  name: string
  version: string
  description: string
}
