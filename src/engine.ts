import {
  getLlama,
  resolveModelFile,
  type Llama,
  type LlamaModel,
  type LlamaEmbeddingContext
} from 'node-llama-cpp'
import { existsSync } from 'node:fs'
import { mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import invariant from 'tiny-invariant'

import { BinaryFileReader } from './binary-file-reader'
import { writeHeader, writeRecord, writeRecords } from './binary-format'
import { CandidateSet } from './candidate-set'
import type { EmbeddingEntry, EngineOptions, SearchResult } from './types'

// Model URI for bge-small-en-v1.5 GGUF (384 dimensions, ~67MB)
const DEFAULT_MODEL_URI =
  'hf:CompendiumLabs/bge-small-en-v1.5-gguf/bge-small-en-v1.5-q8_0.gguf'
const DEFAULT_CACHE_DIR = './.cache/models'

export class EmbeddingEngine {
  private fileReader: BinaryFileReader
  private storePath: string
  private cacheDir: string
  private llama?: Llama
  private model?: LlamaModel
  private embeddingContext?: LlamaEmbeddingContext
  private initPromise?: Promise<void>

  constructor(options: EngineOptions) {
    this.storePath = options.storePath
    this.fileReader = new BinaryFileReader(options.storePath)
    this.cacheDir = options.cacheDir ?? DEFAULT_CACHE_DIR
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
    this.llama = await getLlama()

    const modelPath = await resolveModelFile(DEFAULT_MODEL_URI, this.cacheDir)

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
   * Reads the file in reverse order for efficiency (most recent first)
   * @param key - Unique identifier for the entry
   * @returns The embedding entry, or null if not found
   */
  async get(key: string): Promise<EmbeddingEntry | null> {
    invariant(key, 'Key must be provided.')

    if (!existsSync(this.storePath)) {
      return null
    }

    for await (const entry of this.fileReader.entries()) {
      if (entry.key === key) {
        return entry
      }
    }

    return null
  }

  /**
   * Searches for similar embeddings using cosine similarity
   * @param query - Text query to search for
   * @param limit - Maximum number of results to return (default: 10)
   * @param minSimilarity - Minimum similarity threshold (default: 0, range: -1 to 1)
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

    if (!existsSync(this.storePath)) {
      return []
    }

    const queryEmbedding = await this.generateEmbedding(query)
    const candidateSet = new CandidateSet(limit)

    for await (const entry of this.fileReader.entries()) {
      const similarity = this.cosineSimilarity(queryEmbedding, entry.embedding)

      if (similarity < minSimilarity) {
        continue
      }

      candidateSet.add(entry.key, similarity)
    }

    const results: SearchResult[] = candidateSet.getEntries().map((entry) => ({
      key: entry.key,
      similarity: entry.value
    }))

    return results
  }

  /**
   * Stores a text embedding in the binary append-only file
   * Creates header on first write
   * @param key - Unique identifier for this entry
   * @param text - Text to embed and store
   */
  async store(key: string, text: string): Promise<void> {
    const embedding = await this.generateEmbedding(text)
    const embeddingFloat32 = new Float32Array(embedding)

    const dir = dirname(this.storePath)
    await mkdir(dir, { recursive: true })

    // Write header if file doesn't exist
    if (!existsSync(this.storePath)) {
      await writeHeader(this.storePath, embedding.length)
    }

    // Append record
    await writeRecord(this.storePath, key, embeddingFloat32)
  }

  /**
   * Stores multiple text embeddings in batch
   * More efficient than calling store() multiple times
   * Generates embeddings in parallel and writes all records at once
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

    const dir = dirname(this.storePath)
    await mkdir(dir, { recursive: true })

    // Write header if file doesn't exist
    if (!existsSync(this.storePath)) {
      await writeHeader(this.storePath, embeddingsList[0].length)
    }

    // Prepare records for batch write
    const records = items.map((item, index) => ({
      key: item.key,
      embedding: new Float32Array(embeddingsList[index])
    }))

    await writeRecords(this.storePath, records)
  }

  /**
   * Calculates cosine similarity between two embeddings
   * @param a - First embedding vector
   * @param b - Second embedding vector
   * @returns Cosine similarity score between -1 and 1 (1 = identical, -1 = opposite)
   */
  private cosineSimilarity(a: number[], b: number[]): number {
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
   * Disposes of the cached embedding model and releases resources
   * Call this when you're done using the engine to free up memory
   */
  async dispose(): Promise<void> {
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
