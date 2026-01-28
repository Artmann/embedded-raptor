/**
 * Storage Engine - Main orchestrator for WAL-based embedding storage.
 *
 * Implements the write path: data → fsync → WAL → fsync → index
 * Handles recovery on startup by rebuilding index from WAL.
 */

import { open, stat, mkdir } from 'node:fs/promises'
import type { FileHandle } from 'node:fs/promises'
import { dirname } from 'node:path'
import {
  opType,
  fileExtensions,
  headerSize,
  headerVersionV1
} from './constants'
import {
  serializeDataRecord,
  deserializeDataRecord,
  serializeHeader,
  deserializeHeader
} from './data-format'
import { hashKey } from './wal-format'
import { Wal } from './wal'
import { KeyIndex } from './key-index'
import { FileLock } from './file-lock'
import { Mutex } from './mutex'
import { WriteBatcher } from './write-batcher'
import type {
  DataRecord,
  RecordLocation,
  StorageEngineOptions,
  OpType,
  WriteBatcherOptions
} from './types'

export class StorageEngine {
  private readonly dataPath: string
  private readonly walPath: string
  private readonly lockPath: string
  private readonly dimension: number
  private readonly lockTimeout: number
  private readonly batchingEnabled: boolean
  private readonly batchOptions?: WriteBatcherOptions

  private readonly wal: Wal
  private readonly index: KeyIndex
  private fileLock: FileLock | null = null
  private readonly writeMutex: Mutex
  private writeBatcher: WriteBatcher | null = null

  private dataHandle: FileHandle | null = null
  private dataHandlePromise: Promise<FileHandle> | null = null
  private sequenceCounter: bigint = 0n
  private lockAcquired: boolean = false
  private lockAcquirePromise: Promise<void> | null = null

  private constructor(
    dataPath: string,
    walPath: string,
    lockPath: string,
    dimension: number,
    wal: Wal,
    index: KeyIndex,
    sequenceCounter: bigint,
    lockTimeout: number,
    batchingEnabled: boolean,
    batchOptions?: WriteBatcherOptions
  ) {
    this.dataPath = dataPath
    this.walPath = walPath
    this.lockPath = lockPath
    this.dimension = dimension
    this.wal = wal
    this.index = index
    this.writeMutex = new Mutex()
    this.sequenceCounter = sequenceCounter
    this.lockTimeout = lockTimeout
    this.batchingEnabled = batchingEnabled
    this.batchOptions = batchOptions
  }

  /**
   * Create or open a storage engine.
   * Performs recovery if WAL exists.
   * Lock is NOT acquired at creation time - it's acquired lazily on first write.
   */
  static async create(options: StorageEngineOptions): Promise<StorageEngine> {
    const basePath = options.dataPath.replace(/\.[^.]+$/, '') // Remove extension if present
    const dataPath = basePath + fileExtensions.data
    const walPath = basePath + fileExtensions.wal
    const lockPath = basePath + fileExtensions.lock
    const dimension = options.dimension ?? 384

    // Create WAL instance (not in read-only mode since we may need to write later)
    const wal = new Wal(walPath)

    // Build index from WAL (handles fresh database case)
    const { index, maxSequence } = await KeyIndex.buildFromWal(wal, dataPath)

    return new StorageEngine(
      dataPath,
      walPath,
      lockPath,
      dimension,
      wal,
      index,
      maxSequence + 1n,
      options.lockTimeout ?? 10_000,
      options.batchingEnabled !== false,
      options.batchOptions
    )
  }

  /**
   * Ensures the exclusive lock is acquired before write operations.
   * Called lazily on first write operation.
   */
  private async ensureLockAcquired(): Promise<void> {
    if (this.lockAcquired) {
      return
    }

    // Prevent concurrent lock acquisition
    if (this.lockAcquirePromise) {
      return this.lockAcquirePromise
    }

    this.lockAcquirePromise = this.acquireLockAndInitialize()
    await this.lockAcquirePromise
  }

  private async acquireLockAndInitialize(): Promise<void> {
    // Ensure directory exists
    await mkdir(dirname(this.dataPath), { recursive: true })

    // Check if we need migration from v1
    const needsMigration = await StorageEngine.checkNeedsMigration(this.dataPath)
    if (needsMigration) {
      throw new Error(
        `Database at ${this.dataPath} uses old format (v1). Please run migration first.`
      )
    }

    // Acquire exclusive lock
    this.fileLock = new FileLock(this.lockPath, this.lockTimeout)
    await this.fileLock.acquire()

    try {
      // Create write batcher if enabled (default: true)
      if (this.batchingEnabled) {
        this.writeBatcher = new WriteBatcher(
          this.dataPath,
          this.wal,
          this.index,
          this.dimension,
          this.batchOptions
        )
        await this.writeBatcher.initialize()
      }

      this.lockAcquired = true
    } catch (error) {
      // Release lock on failure
      if (this.fileLock) {
        await this.fileLock.release()
        this.fileLock = null
      }
      throw error
    }
  }

  /**
   * Check if a database file needs migration from v1 format.
   */
  private static async checkNeedsMigration(dataPath: string): Promise<boolean> {
    try {
      const fileHandle = await open(dataPath, 'r')
      try {
        const buffer = new Uint8Array(headerSize)
        await fileHandle.read(buffer, 0, headerSize, 0)
        const header = deserializeHeader(buffer)

        if (header?.version === headerVersionV1) {
          return true
        }
        return false
      } finally {
        await fileHandle.close()
      }
    } catch {
      // File doesn't exist = fresh database
      return false
    }
  }

  /**
   * Write a record to storage.
   * Acquires exclusive lock on first write.
   * Implements: data → fsync → WAL → fsync → index
   */
  async writeRecord(
    key: string,
    embedding: Float32Array,
    op: OpType = opType.insert
  ): Promise<void> {
    // Acquire lock lazily on first write
    await this.ensureLockAcquired()

    if (embedding.length !== this.dimension) {
      throw new Error(
        `Embedding dimension mismatch: expected ${this.dimension}, got ${embedding.length}`
      )
    }

    if (this.writeBatcher) {
      return this.writeRecordBatched(key, embedding, op, this.writeBatcher)
    }

    return this.writeRecordImmediate(key, embedding, op)
  }

  /**
   * Write a record using the batcher for improved throughput.
   */
  private async writeRecordBatched(
    key: string,
    embedding: Float32Array,
    op: OpType,
    batcher: WriteBatcher
  ): Promise<void> {
    await this.writeMutex.acquire()

    let sequenceNumber: bigint
    let recordData: Uint8Array
    let offset: number

    try {
      sequenceNumber = this.sequenceCounter++
      const timestamp = BigInt(Date.now())

      // 1. Serialize data record
      const record: DataRecord = {
        opType: op,
        sequenceNumber,
        timestamp,
        key,
        dimension: this.dimension,
        embedding
      }
      recordData = serializeDataRecord(record)

      // 2. Calculate offset (batcher tracks cumulative file size)
      offset = batcher.calculateNextOffset(recordData.length)
    } finally {
      this.writeMutex.release()
    }

    // 3. Queue write and wait for flush
    return new Promise<void>((resolve, reject) => {
      batcher.queueWrite({
        dataRecord: recordData,
        walEntry: {
          opType: op,
          sequenceNumber,
          offset,
          length: recordData.length,
          keyHash: hashKey(key)
        },
        key,
        op,
        resolve,
        reject
      })
    })
  }

  /**
   * Write a record immediately (non-batched).
   * Implements: data → fsync → WAL → fsync → index
   */
  private async writeRecordImmediate(
    key: string,
    embedding: Float32Array,
    op: OpType
  ): Promise<void> {
    await this.writeMutex.acquire()

    try {
      const sequenceNumber = this.sequenceCounter++
      const timestamp = BigInt(Date.now())

      // 1. Serialize data record
      const record: DataRecord = {
        opType: op,
        sequenceNumber,
        timestamp,
        key,
        dimension: this.dimension,
        embedding
      }
      const recordData = serializeDataRecord(record)

      // 2. Write to data file and fsync
      const offset = await this.appendToDataFile(recordData)

      // 3. Write WAL entry and fsync (COMMIT POINT)
      await this.wal.append({
        opType: op,
        sequenceNumber,
        offset,
        length: recordData.length,
        keyHash: hashKey(key)
      })

      // 4. Update in-memory index
      this.index.apply(
        key,
        {
          offset,
          length: recordData.length,
          sequenceNumber
        },
        op
      )
    } finally {
      this.writeMutex.release()
    }
  }

  /**
   * Read a record by key.
   * O(1) lookup via index.
   */
  async readRecord(key: string): Promise<DataRecord | null> {
    const location = this.index.get(key)
    if (!location) {
      return null
    }

    return this.readRecordAt(location.offset, location.length)
  }

  /**
   * Delete a record by key.
   * Logical delete - writes a delete marker to WAL.
   */
  async deleteRecord(key: string): Promise<boolean> {
    if (!this.index.has(key)) {
      return false
    }

    // Use the same write path as regular writes (supports batching)
    await this.writeRecord(key, new Float32Array(this.dimension), opType.delete)
    return true
  }

  /**
   * Check if a key exists.
   */
  hasKey(key: string): boolean {
    return this.index.has(key)
  }

  /**
   * Get all keys.
   */
  keys(): IterableIterator<string> {
    return this.index.keys()
  }

  /**
   * Iterate over all locations for search.
   */
  locations(): IterableIterator<[string, RecordLocation]> {
    return this.index.locations()
  }

  /**
   * Get the number of records.
   */
  count(): number {
    return this.index.count()
  }

  /**
   * Read the embedding at a specific offset (for search optimization).
   */
  async readEmbeddingAt(offset: number): Promise<Float32Array | null> {
    const dataHandle = await this.getDataHandle()

    // Calculate where embedding starts in record
    // magic(4) + version(2) + opType(1) + flags(1) + seqNum(8) + timestamp(8) + keyLen(2) = 26
    // Then key (variable), then dimension(4), then embedding
    // We need to read keyLen first to know the offset

    const headerBuffer = new Uint8Array(28) // Read up to keyLen + 2 bytes
    await dataHandle.read(headerBuffer, 0, 28, offset)

    const keyLen = new DataView(headerBuffer.buffer).getUint16(24, true)
    const embeddingOffset = offset + 26 + keyLen + 4

    const embeddingBuffer = new Uint8Array(this.dimension * 4)
    await dataHandle.read(
      embeddingBuffer,
      0,
      this.dimension * 4,
      embeddingOffset
    )

    return new Float32Array(embeddingBuffer.buffer)
  }

  /**
   * Get the embedding dimension.
   */
  getDimension(): number {
    return this.dimension
  }

  /**
   * Force flush all pending writes to disk.
   * Only needed when using batching and you need explicit durability.
   */
  async flush(): Promise<void> {
    if (this.writeBatcher) {
      await this.writeBatcher.forceFlush()
    }
  }

  /**
   * Close the storage engine.
   */
  async close(): Promise<void> {
    // Flush and close the batcher first
    if (this.writeBatcher) {
      await this.writeBatcher.close()
    }

    if (this.dataHandle) {
      await this.dataHandle.close()
      this.dataHandle = null
    }
    await this.wal.close()

    // Release lock if we acquired one
    if (this.fileLock) {
      await this.fileLock.release()
    }
  }

  /**
   * Check if the storage engine has acquired the write lock.
   */
  hasWriteLock(): boolean {
    return this.lockAcquired
  }

  /**
   * Append data to the data file.
   */
  private async appendToDataFile(data: Uint8Array): Promise<number> {
    const dataHandle = await this.getDataHandleForWrite()

    // Get current file size (this is where we'll append)
    const stats = await stat(this.dataPath).catch(() => ({ size: 0 }))
    let offset = stats.size

    // If file is empty, write header first
    if (offset === 0) {
      const header = serializeHeader(this.dimension)
      await dataHandle.write(header, 0, header.length, 0)
      await dataHandle.sync()
      offset = headerSize
    }

    // Append record
    await dataHandle.write(data, 0, data.length, offset)
    await dataHandle.sync()

    return offset
  }

  /**
   * Read a record at a specific offset.
   */
  private async readRecordAt(
    offset: number,
    length: number
  ): Promise<DataRecord | null> {
    const dataHandle = await this.getDataHandle()

    const buffer = new Uint8Array(length)
    await dataHandle.read(buffer, 0, length, offset)

    const result = deserializeDataRecord(buffer)
    return result?.record ?? null
  }

  /**
   * Get or open the data file handle.
   * Uses promise-based locking to prevent race conditions in concurrent access.
   * Opens in read mode for reads, upgraded to read-write when lock is acquired.
   */
  private async getDataHandle(): Promise<FileHandle> {
    if (this.dataHandle) {
      return this.dataHandle
    }

    // Use promise-based lock to ensure only one open operation runs
    this.dataHandlePromise ??= (async () => {
      // Try to open for reading first (works for read operations)
      const handle = await open(this.dataPath, 'r').catch(async () => {
        // File doesn't exist - only create if we have the lock
        if (this.lockAcquired) {
          await mkdir(dirname(this.dataPath), { recursive: true })
          return open(this.dataPath, 'w+')
        }
        throw new Error(`Database file does not exist: ${this.dataPath}`)
      })
      this.dataHandle = handle
      return handle
    })()

    return this.dataHandlePromise
  }

  /**
   * Get or open the data file handle for writing.
   * Creates the file if it doesn't exist.
   */
  private async getDataHandleForWrite(): Promise<FileHandle> {
    // Close any read-only handle first
    if (this.dataHandle && !this.lockAcquired) {
      await this.dataHandle.close()
      this.dataHandle = null
      this.dataHandlePromise = null
    }

    if (this.dataHandle) {
      return this.dataHandle
    }

    // Use promise-based lock to ensure only one open operation runs
    this.dataHandlePromise ??= (async () => {
      // Ensure directory exists
      await mkdir(dirname(this.dataPath), { recursive: true })
      const handle = await open(this.dataPath, 'r+').catch(async () => {
        // File doesn't exist, create it
        return open(this.dataPath, 'w+')
      })
      this.dataHandle = handle
      return handle
    })()

    return this.dataHandlePromise
  }
}
