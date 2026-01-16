/**
 * Types for write batching.
 */

import type { WalEntry, OpType } from './types'

/**
 * A pending write operation waiting to be flushed.
 */
export interface PendingWrite {
  /** Serialized data record to write to data file */
  dataRecord: Uint8Array
  /** The WAL entry to write */
  walEntry: WalEntry
  /** Key for index update */
  key: string
  /** Operation type for index update */
  op: OpType
  /** Resolve function to call on successful flush */
  resolve: () => void
  /** Reject function to call on failure */
  reject: (error: Error) => void
}

/**
 * Options for configuring the write batcher.
 */
export interface WriteBatcherOptions {
  /** Maximum number of writes to batch before flushing (default: 100) */
  maxBatchSize?: number
  /** Maximum time in ms to wait before flushing (default: 10) */
  maxBatchDelayMs?: number
  /** Maximum total bytes in data file writes before flushing (default: 1MB) */
  maxBatchBytes?: number
}

/**
 * Result of a batch flush operation.
 */
export interface FlushResult {
  /** Number of writes successfully flushed */
  count: number
  /** Total bytes written to data file */
  dataBytes: number
  /** Total bytes written to WAL */
  walBytes: number
}
