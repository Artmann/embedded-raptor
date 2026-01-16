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
  OpType
} from './types'

export class StorageEngine {
  private readonly dataPath: string
  private readonly walPath: string
  private readonly lockPath: string
  private readonly dimension: number

  private readonly wal: Wal
  private readonly index: KeyIndex
  private readonly fileLock: FileLock
  private readonly writeMutex: Mutex
  private readonly writeBatcher: WriteBatcher | null

  private dataHandle: FileHandle | null = null
  private sequenceCounter: bigint = 0n

  private constructor(
    dataPath: string,
    walPath: string,
    lockPath: string,
    dimension: number,
    wal: Wal,
    index: KeyIndex,
    fileLock: FileLock,
    sequenceCounter: bigint,
    writeBatcher: WriteBatcher | null
  ) {
    this.dataPath = dataPath
    this.walPath = walPath
    this.lockPath = lockPath
    this.dimension = dimension
    this.wal = wal
    this.index = index
    this.fileLock = fileLock
    this.writeMutex = new Mutex()
    this.sequenceCounter = sequenceCounter
    this.writeBatcher = writeBatcher
  }

  /**
   * Create or open a storage engine.
   * Performs recovery if WAL exists.
   */
  static async create(options: StorageEngineOptions): Promise<StorageEngine> {
    const basePath = options.dataPath.replace(/\.[^.]+$/, '') // Remove extension if present
    const dataPath = basePath + fileExtensions.data
    const walPath = basePath + fileExtensions.wal
    const lockPath = basePath + fileExtensions.lock
    const dimension = options.dimension ?? 384

    // Ensure directory exists
    await mkdir(dirname(dataPath), { recursive: true })

    // Acquire exclusive lock
    const fileLock = new FileLock(lockPath, options.lockTimeout)
    await fileLock.acquire()

    try {
      // Check if we need migration from v1
      const needsMigration = await StorageEngine.checkNeedsMigration(dataPath)
      if (needsMigration) {
        // Migration will be handled separately
        throw new Error(
          `Database at ${dataPath} uses old format (v1). Please run migration first.`
        )
      }

      // Create WAL instance
      const wal = new Wal(walPath)

      // Build index from WAL (handles fresh database case)
      const { index, maxSequence } = await KeyIndex.buildFromWal(wal, dataPath)

      // Create write batcher if enabled (default: true)
      let writeBatcher: WriteBatcher | null = null
      if (options.batchingEnabled !== false) {
        writeBatcher = new WriteBatcher(
          dataPath,
          wal,
          index,
          dimension,
          options.batchOptions
        )
        await writeBatcher.initialize()
      }

      return new StorageEngine(
        dataPath,
        walPath,
        lockPath,
        dimension,
        wal,
        index,
        fileLock,
        maxSequence + 1n,
        writeBatcher
      )
    } catch (error) {
      await fileLock.release()
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
   * Implements: data → fsync → WAL → fsync → index
   */
  async writeRecord(
    key: string,
    embedding: Float32Array,
    op: OpType = opType.insert
  ): Promise<void> {
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
    await this.fileLock.release()
  }

  /**
   * Append data to the data file.
   */
  private async appendToDataFile(data: Uint8Array): Promise<number> {
    const dataHandle = await this.getDataHandle()

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
   */
  private async getDataHandle(): Promise<FileHandle> {
    if (!this.dataHandle) {
      // Ensure directory exists
      await mkdir(dirname(this.dataPath), { recursive: true })
      this.dataHandle = await open(this.dataPath, 'r+').catch(async () => {
        // File doesn't exist, create it
        return open(this.dataPath, 'w+')
      })
    }
    return this.dataHandle
  }
}
