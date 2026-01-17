import {
  getLlama,
  LlamaLogLevel,
  resolveModelFile,
  type Llama,
  type LlamaModel,
  type LlamaEmbeddingContext
} from 'node-llama-cpp'
import invariant from 'tiny-invariant'

import {
  StorageEngine,
  ensureV2Format,
  opType,
  ReadOnlyError
} from './storage-engine'
import { CandidateSet } from './candidate-set'
import { LRUCache } from './lru-cache'
import type { EmbeddingEntry, EngineOptions, SearchResult } from './types'

// Model URI for bge-small-en-v1.5 GGUF (384 dimensions, ~67MB)
const defaultModelUri =
  'hf:CompendiumLabs/bge-small-en-v1.5-gguf/bge-small-en-v1.5-q8_0.gguf'
const defaultCacheDir = './.cache/models'
const defaultDimension = 384

// Static set of engines with loaded native resources, used by shared exit handler
const enginesWithNativeResources = new Set<EmbeddingEngine>()
let exitHandlerRegistered = false

/**
 * Shared exit handler that nullifies native references for all active engines.
 * This prevents segfaults when process.exit() is called without disposing engines.
 * Uses a single listener to avoid adding multiple exit handlers.
 */
function handleProcessExit(): void {
  for (const engine of enginesWithNativeResources) {
    engine._nullifyNativeResources()
  }
  enginesWithNativeResources.clear()
}

export class EmbeddingEngine {
  private storageEngine: StorageEngine | null = null
  private storePath: string
  private cacheDir: string
  private dimension: number
  private readonly readOnly: boolean
  private llama?: Llama
  private model?: LlamaModel
  private embeddingContext?: LlamaEmbeddingContext
  private initPromise?: Promise<void>
  private storageInitPromise?: Promise<StorageEngine>
  // In-memory embedding cache for faster search
  private embeddingCache: Map<string, Float32Array> | null = null
  // LRU cache for text-to-embedding lookups (avoids regenerating embeddings for repeated text)
  private textEmbeddingCache: LRUCache<string, Float32Array> | null = null

  constructor(options: EngineOptions) {
    this.storePath = options.storePath
    this.cacheDir = options.cacheDir ?? defaultCacheDir
    this.dimension = defaultDimension
    this.readOnly = options.readOnly ?? false
    if (options.embeddingCacheSize && options.embeddingCacheSize > 0) {
      this.textEmbeddingCache = new LRUCache(options.embeddingCacheSize)
    }
  }

  /**
   * Gets or initializes the storage engine
   * Performs migration from v1 format if needed
   */
  private async ensureStorageEngine(): Promise<StorageEngine> {
    if (this.storageEngine) {
      return this.storageEngine
    }

    // Prevent concurrent initialization
    if (this.storageInitPromise) {
      return this.storageInitPromise
    }

    this.storageInitPromise = this.initializeStorage()
    this.storageEngine = await this.storageInitPromise
    return this.storageEngine
  }

  private async initializeStorage(): Promise<StorageEngine> {
    // Check and migrate from v1 format if needed (skip for read-only mode)
    if (!this.readOnly) {
      await ensureV2Format(this.storePath, this.dimension)
    }

    // Create storage engine
    return StorageEngine.create({
      dataPath: this.storePath,
      dimension: this.dimension,
      readOnly: this.readOnly
    })
  }

  /**
   * Loads all embeddings into memory for faster search
   * Called lazily on first search
   * @returns The embedding cache Map
   */
  private async ensureEmbeddingCache(
    storage: StorageEngine
  ): Promise<Map<string, Float32Array>> {
    if (this.embeddingCache !== null) {
      return this.embeddingCache
    }

    this.embeddingCache = new Map()

    for (const [key, location] of storage.locations()) {
      const embedding = await storage.readEmbeddingAt(location.offset)
      if (embedding) {
        this.embeddingCache.set(key, embedding)
      }
    }

    return this.embeddingCache
  }

  /**
   * Invalidates the embedding cache, forcing reload on next search
   */
  private invalidateEmbeddingCache(): void {
    this.embeddingCache = null
  }

  /**
   * Gets or initializes the embedding model
   * Caches the model instance to avoid repeated initialization overhead
   */
  private async ensureModelLoaded(): Promise<void> {
    if (this.embeddingContext) {
      return
    }

    // Prevent concurrent initialization
    if (this.initPromise) {
      return this.initPromise
    }

    this.initPromise = this.initializeModel()
    await this.initPromise
  }

  private async initializeModel(): Promise<void> {
    this.llama = await getLlama({
      logLevel: LlamaLogLevel.error // Suppress tokenizer warnings for embedding models
    })

    const modelPath = await resolveModelFile(defaultModelUri, this.cacheDir)

    this.model = await this.llama.loadModel({
      modelPath
    })

    this.embeddingContext = await this.model.createEmbeddingContext()

    // Register this engine for cleanup on process exit
    enginesWithNativeResources.add(this)
    if (!exitHandlerRegistered) {
      process.on('exit', handleProcessExit)
      exitHandlerRegistered = true
    }
  }

  /**
   * Truncates text to fit within the model's context size
   * Uses the model's tokenizer for accurate token counting
   * BGE-small supports 512 tokens, we use 500 to leave room for special tokens
   */
  private truncateToContextSize(text: string): string {
    if (!this.model) {
      // Fallback if model not loaded yet
      const maxChars = 300 * 3
      return text.length <= maxChars ? text : text.slice(0, maxChars)
    }

    const maxTokens = 500
    const tokens = this.model.tokenize(text)

    if (tokens.length <= maxTokens) {
      return text
    }

    // Truncate tokens and detokenize
    const truncatedTokens = tokens.slice(0, maxTokens)
    return this.model.detokenize(truncatedTokens)
  }

  /**
   * Generates embedding from text using node-llama-cpp with bge-small-en-v1.5 model
   * @param text - Text to embed
   * @returns 384-dimensional embedding vector (normalized)
   */
  async generateEmbedding(text: string): Promise<number[]> {
    const embedding = await this.generateEmbeddingFloat32(text)
    return Array.from(embedding)
  }

  /**
   * Internal method that returns embedding as Float32Array for performance
   * Uses Float32Array throughout internal operations to avoid boxing overhead
   * Checks the text embedding cache first to avoid regenerating embeddings
   */
  private async generateEmbeddingFloat32(text: string): Promise<Float32Array> {
    // Check cache first
    if (this.textEmbeddingCache) {
      const cached = this.textEmbeddingCache.get(text)
      if (cached) {
        return cached
      }
    }

    await this.ensureModelLoaded()
    invariant(this.embeddingContext, 'Embedding context not initialized')

    const truncatedText = this.truncateToContextSize(text)
    const embedding = await this.embeddingContext.getEmbeddingFor(truncatedText)
    const result = new Float32Array(embedding.vector)

    // Store in cache
    if (this.textEmbeddingCache) {
      this.textEmbeddingCache.set(text, result)
    }

    return result
  }

  /**
   * Retrieves an embedding entry by key
   * O(1) lookup via in-memory index
   * @param key - Unique identifier for the entry
   * @returns The embedding entry, or null if not found
   */
  async get(key: string): Promise<EmbeddingEntry | null> {
    invariant(key, 'Key must be provided.')

    const storage = await this.ensureStorageEngine()
    const record = await storage.readRecord(key)

    if (!record) {
      return null
    }

    return {
      key: record.key,
      text: '', // Text is not stored in v2 format
      embedding: Array.from(record.embedding),
      timestamp: Number(record.timestamp)
    }
  }

  /**
   * Checks if a key exists in the database
   * O(1) lookup via in-memory index
   * @param key - Unique identifier for the entry
   * @returns true if the key exists, false otherwise
   */
  async has(key: string): Promise<boolean> {
    invariant(key, 'Key must be provided.')

    const storage = await this.ensureStorageEngine()
    return storage.hasKey(key)
  }

  /**
   * Searches for similar embeddings using cosine similarity
   * @param query - Text query to search for
   * @param limit - Maximum number of results to return (default: 10)
   * @param minSimilarity - Minimum similarity threshold (default: 0.5, range: 0 to 1)
   * @returns Array of search results sorted by similarity (highest first)
   */
  async search(
    query: string,
    limit: number = 10,
    minSimilarity: number = 0.5
  ): Promise<SearchResult[]> {
    invariant(query, 'Query text must be provided.')
    invariant(limit > 0, 'Limit must be a positive integer.')
    invariant(
      minSimilarity >= 0 && minSimilarity <= 1,
      'minSimilarity must be between 0 and 1.'
    )

    const storage = await this.ensureStorageEngine()

    if (storage.count() === 0) {
      return []
    }

    const queryEmbedding = await this.generateEmbeddingFloat32(query)
    const candidateSet = new CandidateSet(limit)

    // Load embeddings into cache if not already cached
    const cache = await this.ensureEmbeddingCache(storage)

    // Iterate through all entries using cached embeddings
    for (const key of storage.keys()) {
      const embedding = cache.get(key)
      if (!embedding) {
        continue
      }

      const similarity = this.cosineSimilarity(queryEmbedding, embedding)

      if (similarity < minSimilarity) {
        continue
      }

      candidateSet.add(key, similarity)
    }

    const results: SearchResult[] = candidateSet.getEntries().map((entry) => ({
      key: entry.key,
      similarity: entry.value
    }))

    return results
  }

  /**
   * Stores a text embedding with WAL-based durability
   * @param key - Unique identifier for this entry
   * @param text - Text to embed and store
   */
  async store(key: string, text: string): Promise<void> {
    if (this.readOnly) {
      throw new ReadOnlyError()
    }
    invariant(key, 'Key must be provided.')
    invariant(text, 'Text must be provided.')

    const embedding = await this.generateEmbeddingFloat32(text)
    const storage = await this.ensureStorageEngine()

    // Determine if this is an insert or update
    const op = storage.hasKey(key) ? opType.update : opType.insert
    await storage.writeRecord(key, embedding, op)

    // Update cache if it exists (avoid invalidating entire cache)
    if (this.embeddingCache !== null) {
      this.embeddingCache.set(key, embedding)
    }
  }

  /**
   * Stores multiple text embeddings in batch
   * More efficient than calling store() multiple times
   * Generates embeddings in parallel and writes records sequentially
   * Uses text embedding cache to avoid regenerating embeddings for duplicate texts
   * @param items - Array of {key, text} objects to store
   */
  async storeMany(items: Array<{ key: string; text: string }>): Promise<void> {
    if (this.readOnly) {
      throw new ReadOnlyError()
    }
    invariant(items.length > 0, 'Items array must not be empty.')

    await this.ensureModelLoaded()
    const embeddingContext = this.embeddingContext
    invariant(embeddingContext, 'Embedding context not initialized')

    // Generate embeddings in parallel, using cache when available
    const embeddingPromises = items.map(async (item) => {
      // Check text embedding cache first
      if (this.textEmbeddingCache) {
        const cached = this.textEmbeddingCache.get(item.text)
        if (cached) {
          return cached
        }
      }

      const truncatedText = this.truncateToContextSize(item.text)
      const embedding = await embeddingContext.getEmbeddingFor(truncatedText)
      const result = new Float32Array(embedding.vector)

      // Store in text embedding cache
      if (this.textEmbeddingCache) {
        this.textEmbeddingCache.set(item.text, result)
      }

      return result
    })

    const embeddingsList = await Promise.all(embeddingPromises)

    const storage = await this.ensureStorageEngine()

    // Determine operation types before concurrent writes
    const operations = items.map((item) => ({
      key: item.key,
      op: storage.hasKey(item.key) ? opType.update : opType.insert
    }))

    // Write records concurrently (batching will group fsyncs)
    const writePromises = items.map((item, i) => {
      const embedding = embeddingsList[i]
      return storage.writeRecord(item.key, embedding, operations[i].op)
    })

    await Promise.all(writePromises)

    // Update cache after all writes complete
    if (this.embeddingCache !== null) {
      for (let i = 0; i < items.length; i++) {
        this.embeddingCache.set(items[i].key, embeddingsList[i])
      }
    }
  }

  /**
   * Deletes an entry by key
   * Logical delete - records a delete marker in the WAL
   * @param key - Unique identifier for the entry to delete
   * @returns true if the entry was deleted, false if it didn't exist
   */
  async delete(key: string): Promise<boolean> {
    if (this.readOnly) {
      throw new ReadOnlyError()
    }
    invariant(key, 'Key must be provided.')

    const storage = await this.ensureStorageEngine()
    const deleted = await storage.deleteRecord(key)

    // Remove from cache if it exists
    if (deleted && this.embeddingCache !== null) {
      this.embeddingCache.delete(key)
    }

    return deleted
  }

  /**
   * Gets all keys in the database
   * @returns Iterator of all keys
   */
  async keys(): Promise<string[]> {
    const storage = await this.ensureStorageEngine()
    return Array.from(storage.keys())
  }

  /**
   * Gets the number of entries in the database
   * @returns Number of entries
   */
  async count(): Promise<number> {
    const storage = await this.ensureStorageEngine()
    return storage.count()
  }

  /**
   * Calculates cosine similarity between two Float32Arrays
   * Uses typed arrays throughout to avoid boxing overhead
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      throw new Error('Embeddings must have the same dimensions')
    }

    let dotProduct = 0
    let magnitudeA = 0
    let magnitudeB = 0

    for (let i = 0; i < a.length; i++) {
      const ai = a[i]
      const bi = b[i]
      dotProduct += ai * bi
      magnitudeA += ai * ai
      magnitudeB += bi * bi
    }

    magnitudeA = Math.sqrt(magnitudeA)
    magnitudeB = Math.sqrt(magnitudeB)

    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0
    }

    return dotProduct / (magnitudeA * magnitudeB)
  }

  /**
   * Check if the engine is in read-only mode.
   */
  isReadOnly(): boolean {
    return this.readOnly
  }

  /**
   * Gets statistics about the text embedding cache.
   * @returns Cache stats if enabled, null if cache is disabled
   */
  getTextEmbeddingCacheStats(): { size: number; maxSize: number } | null {
    if (!this.textEmbeddingCache) {
      return null
    }
    return {
      size: this.textEmbeddingCache.size(),
      maxSize: this.textEmbeddingCache.getMaxSize()
    }
  }

  /**
   * Internal method called by the shared exit handler to nullify native references.
   * @internal
   */
  _nullifyNativeResources(): void {
    this.embeddingContext = undefined
    this.model = undefined
    this.llama = undefined
  }

  /**
   * Disposes of resources and closes the storage engine
   * Call this when you're done using the engine to free up memory
   */
  async dispose(): Promise<void> {
    // Remove from shared exit handler tracking
    enginesWithNativeResources.delete(this)

    // Clear embedding cache
    this.embeddingCache = null

    // Clear text embedding cache
    if (this.textEmbeddingCache) {
      this.textEmbeddingCache.clear()
      this.textEmbeddingCache = null
    }

    // Close storage engine
    if (this.storageEngine) {
      await this.storageEngine.close()
      this.storageEngine = null
    }
    this.storageInitPromise = undefined

    // Dispose embedding model
    if (this.embeddingContext) {
      await this.embeddingContext.dispose()
      this.embeddingContext = undefined
    }
    if (this.model) {
      await this.model.dispose()
      this.model = undefined
    }
    this.llama = undefined
    this.initPromise = undefined
  }
}
