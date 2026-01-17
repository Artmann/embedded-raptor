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
}

export interface PackageJson {
  name: string
  version: string
  description: string
}
