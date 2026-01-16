/**
 * Write Batcher - Batches multiple writes and flushes them together.
 *
 * This reduces fsync overhead by batching writes and doing a single
 * fsync for the entire batch instead of one per write.
 */

import { open, stat, mkdir } from 'node:fs/promises'
import type { FileHandle } from 'node:fs/promises'
import { dirname } from 'node:path'
import { Mutex } from './mutex'
import type { Wal } from './wal'
import type { KeyIndex } from './key-index'
import { headerSize } from './constants'
import { serializeHeader } from './data-format'
import type {
  PendingWrite,
  WriteBatcherOptions,
  FlushResult
} from './write-batch'

const DEFAULT_MAX_BATCH_SIZE = 100
const DEFAULT_MAX_BATCH_DELAY_MS = 10
const DEFAULT_MAX_BATCH_BYTES = 1024 * 1024 // 1MB

export class WriteBatcher {
  private readonly dataPath: string
  private readonly wal: Wal
  private readonly index: KeyIndex
  private readonly dimension: number
  private readonly maxBatchSize: number
  private readonly maxBatchDelayMs: number
  private readonly maxBatchBytes: number

  private dataHandle: FileHandle | null = null
  private pendingWrites: PendingWrite[] = []
  private flushScheduled: boolean = false
  private currentBatchBytes: number = 0
  private flushMutex: Mutex = new Mutex()
  private closed: boolean = false
  private currentDataFileSize: number = 0

  constructor(
    dataPath: string,
    wal: Wal,
    index: KeyIndex,
    dimension: number,
    options?: WriteBatcherOptions
  ) {
    this.dataPath = dataPath
    this.wal = wal
    this.index = index
    this.dimension = dimension
    this.maxBatchSize = options?.maxBatchSize ?? DEFAULT_MAX_BATCH_SIZE
    this.maxBatchDelayMs =
      options?.maxBatchDelayMs ?? DEFAULT_MAX_BATCH_DELAY_MS
    this.maxBatchBytes = options?.maxBatchBytes ?? DEFAULT_MAX_BATCH_BYTES
  }

  /**
   * Initialize the batcher by getting current data file size.
   */
  async initialize(): Promise<void> {
    try {
      const stats = await stat(this.dataPath)
      this.currentDataFileSize = stats.size
    } catch {
      this.currentDataFileSize = 0
    }
  }

  /**
   * Calculate the offset for a new write.
   * Must be called under external synchronization (the storage engine's writeMutex).
   */
  calculateNextOffset(dataLength: number): number {
    let offset: number

    if (this.currentDataFileSize === 0) {
      // First write - account for header
      offset = headerSize
      this.currentDataFileSize = headerSize + dataLength
    } else {
      offset = this.currentDataFileSize
      this.currentDataFileSize += dataLength
    }

    return offset
  }

  /**
   * Queue a write operation for batching.
   */
  queueWrite(write: PendingWrite): void {
    if (this.closed) {
      write.reject(new Error('WriteBatcher is closed'))
      return
    }

    this.pendingWrites.push(write)
    this.currentBatchBytes += write.dataRecord.length

    // Check if we should flush immediately based on size thresholds
    if (this.shouldFlushNow()) {
      this.triggerFlush()
    } else {
      // Schedule flush - use short delay to batch concurrent writes
      // without penalizing sequential writes
      this.scheduleFlush()
    }
  }

  private shouldFlushNow(): boolean {
    return (
      this.pendingWrites.length >= this.maxBatchSize ||
      this.currentBatchBytes >= this.maxBatchBytes
    )
  }

  private scheduleFlush(): void {
    if (!this.flushScheduled) {
      this.flushScheduled = true
      // Use setImmediate to batch concurrent writes in the same tick
      // without adding delay for sequential writes
      setImmediate(() => {
        this.triggerFlush()
      })
    }
  }

  private triggerFlush(): void {
    this.flushScheduled = false

    // Fire and forget - errors handled in flush()
    this.flush().catch(() => {
      // Errors are propagated to individual writes
    })
  }

  /**
   * Flush all pending writes to disk.
   */
  private async flush(): Promise<FlushResult> {
    await this.flushMutex.acquire()

    try {
      // Grab current batch and reset
      const writes = this.pendingWrites
      const batchBytes = this.currentBatchBytes
      this.pendingWrites = []
      this.currentBatchBytes = 0

      if (writes.length === 0) {
        return { count: 0, dataBytes: 0, walBytes: 0 }
      }

      try {
        // 1. Write all data records to data file + fsync
        await this.writeDataBatch(writes)

        // 2. Write all WAL entries and fsync (COMMIT POINT)
        await this.wal.appendBatch(writes.map((w) => w.walEntry))

        // 3. Update index for all writes
        for (const write of writes) {
          this.index.apply(
            write.key,
            {
              offset: write.walEntry.offset,
              length: write.walEntry.length,
              sequenceNumber: write.walEntry.sequenceNumber
            },
            write.op
          )
        }

        // 4. Resolve all promises
        for (const write of writes) {
          write.resolve()
        }

        return {
          count: writes.length,
          dataBytes: batchBytes,
          walBytes: writes.length * 48
        }
      } catch (error) {
        // On any failure, reject all writes in this batch
        const err = error instanceof Error ? error : new Error(String(error))
        for (const write of writes) {
          write.reject(err)
        }

        // Re-throw to signal flush failure
        throw error
      }
    } finally {
      this.flushMutex.release()
    }
  }

  private async writeDataBatch(writes: PendingWrite[]): Promise<void> {
    const dataHandle = await this.getDataHandle()

    // Determine write start offset
    const firstOffset = writes[0].walEntry.offset
    let writeOffset = firstOffset

    // If first record starts at header, we need to write header first
    if (firstOffset === headerSize) {
      const header = serializeHeader(this.dimension)
      await dataHandle.write(header, 0, header.length, 0)
      writeOffset = headerSize
    }

    // Calculate total buffer size needed
    const totalDataSize = writes.reduce(
      (sum, w) => sum + w.dataRecord.length,
      0
    )

    // Combine all data records into single buffer for single write
    const combinedBuffer = new Uint8Array(totalDataSize)
    let bufferOffset = 0

    for (const write of writes) {
      combinedBuffer.set(write.dataRecord, bufferOffset)
      bufferOffset += write.dataRecord.length
    }

    // Single write for all data
    await dataHandle.write(
      combinedBuffer,
      0,
      combinedBuffer.length,
      writeOffset
    )

    // Single fsync for data file
    await dataHandle.sync()
  }

  private async getDataHandle(): Promise<FileHandle> {
    if (!this.dataHandle) {
      await mkdir(dirname(this.dataPath), { recursive: true })
      this.dataHandle = await open(this.dataPath, 'r+').catch(async () => {
        return open(this.dataPath, 'w+')
      })
    }
    return this.dataHandle
  }

  /**
   * Force flush all pending writes immediately.
   */
  async forceFlush(): Promise<FlushResult> {
    this.flushScheduled = false
    return this.flush()
  }

  /**
   * Close the batcher, flushing any pending writes.
   */
  async close(): Promise<void> {
    this.closed = true

    // Flush any remaining writes
    await this.forceFlush()

    // Close file handle
    if (this.dataHandle) {
      await this.dataHandle.close()
      this.dataHandle = null
    }
  }
}
