import {
  getLlama,
  LlamaLogLevel,
  resolveModelFile,
  type Llama,
  type LlamaModel,
  type LlamaEmbeddingContext
} from 'node-llama-cpp'
import invariant from 'tiny-invariant'

import { StorageEngine, ensureV2Format, opType } from './storage-engine'
import { CandidateSet } from './candidate-set'
import type { EmbeddingEntry, EngineOptions, SearchResult } from './types'

// Model URI for bge-small-en-v1.5 GGUF (384 dimensions, ~67MB)
const defaultModelUri =
  'hf:CompendiumLabs/bge-small-en-v1.5-gguf/bge-small-en-v1.5-q8_0.gguf'
const defaultCacheDir = './.cache/models'
const defaultDimension = 384

export class EmbeddingEngine {
  private storageEngine: StorageEngine | null = null
  private storePath: string
  private cacheDir: string
  private dimension: number
  private llama?: Llama
  private model?: LlamaModel
  private embeddingContext?: LlamaEmbeddingContext
  private initPromise?: Promise<void>
  private storageInitPromise?: Promise<StorageEngine>

  constructor(options: EngineOptions) {
    this.storePath = options.storePath
    this.cacheDir = options.cacheDir ?? defaultCacheDir
    this.dimension = defaultDimension
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
    // Check and migrate from v1 format if needed
    await ensureV2Format(this.storePath, this.dimension)

    // Create storage engine
    return StorageEngine.create({
      dataPath: this.storePath,
      dimension: this.dimension
    })
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
    await this.ensureModelLoaded()
    invariant(this.embeddingContext, 'Embedding context not initialized')

    const truncatedText = this.truncateToContextSize(text)
    const embedding = await this.embeddingContext.getEmbeddingFor(truncatedText)

    return Array.from(embedding.vector)
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
      timestamp: Number(record.sequenceNumber)
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

    const queryEmbedding = await this.generateEmbedding(query)
    const candidateSet = new CandidateSet(limit)

    // Iterate through all entries in the index
    for (const [key, location] of storage.locations()) {
      // Read embedding directly from data file
      const embedding = await storage.readEmbeddingAt(location.offset)
      if (!embedding) {
        continue
      }

      const similarity = this.cosineSimilarityFloat32(queryEmbedding, embedding)

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
    invariant(key, 'Key must be provided.')
    invariant(text, 'Text must be provided.')

    const embedding = await this.generateEmbedding(text)
    const embeddingFloat32 = new Float32Array(embedding)

    const storage = await this.ensureStorageEngine()

    // Determine if this is an insert or update
    const op = storage.hasKey(key) ? opType.update : opType.insert
    await storage.writeRecord(key, embeddingFloat32, op)
  }

  /**
   * Stores multiple text embeddings in batch
   * More efficient than calling store() multiple times
   * Generates embeddings in parallel and writes records sequentially
   * @param items - Array of {key, text} objects to store
   */
  async storeMany(items: Array<{ key: string; text: string }>): Promise<void> {
    invariant(items.length > 0, 'Items array must not be empty.')

    await this.ensureModelLoaded()
    const embeddingContext = this.embeddingContext
    invariant(embeddingContext, 'Embedding context not initialized')

    // Generate embeddings in parallel
    const embeddingPromises = items.map(async (item) => {
      const truncatedText = this.truncateToContextSize(item.text)
      const embedding = await embeddingContext.getEmbeddingFor(truncatedText)
      return Array.from(embedding.vector)
    })

    const embeddingsList = await Promise.all(embeddingPromises)

    const storage = await this.ensureStorageEngine()

    // Write records sequentially (storage engine handles locking)
    for (let i = 0; i < items.length; i++) {
      const key = items[i].key
      const embedding = new Float32Array(embeddingsList[i])
      const op = storage.hasKey(key) ? opType.update : opType.insert
      await storage.writeRecord(key, embedding, op)
    }
  }

  /**
   * Deletes an entry by key
   * Logical delete - records a delete marker in the WAL
   * @param key - Unique identifier for the entry to delete
   * @returns true if the entry was deleted, false if it didn't exist
   */
  async delete(key: string): Promise<boolean> {
    invariant(key, 'Key must be provided.')

    const storage = await this.ensureStorageEngine()
    return storage.deleteRecord(key)
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
   * Calculates cosine similarity between number[] and Float32Array
   */
  private cosineSimilarityFloat32(a: number[], b: Float32Array): number {
    if (a.length !== b.length) {
      throw new Error('Embeddings must have the same dimensions')
    }

    let dotProduct = 0
    let magnitudeA = 0
    let magnitudeB = 0

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i]
      magnitudeA += a[i] * a[i]
      magnitudeB += b[i] * b[i]
    }

    magnitudeA = Math.sqrt(magnitudeA)
    magnitudeB = Math.sqrt(magnitudeB)

    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0
    }

    return dotProduct / (magnitudeA * magnitudeB)
  }

  /**
   * Disposes of resources and closes the storage engine
   * Call this when you're done using the engine to free up memory
   */
  async dispose(): Promise<void> {
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
