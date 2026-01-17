/**
 * Types for the WAL-based storage engine.
 */

import type { opType } from './constants'

/**
 * Operation type for storage operations.
 */
export type OpType = (typeof opType)[keyof typeof opType]

/**
 * Location of a record in the data file.
 * Used by the in-memory index for O(1) lookups.
 */
export interface RecordLocation {
  /** Byte offset in the data file */
  offset: number
  /** Length of the record in bytes */
  length: number
  /** Sequence number for ordering */
  sequenceNumber: bigint
}

/**
 * WAL entry structure.
 * Fixed 48 bytes on disk, points to data file location.
 */
export interface WalEntry {
  /** Operation type (INSERT, UPDATE, DELETE) */
  opType: OpType
  /** Monotonic sequence number */
  sequenceNumber: bigint
  /** Offset in data file where record starts */
  offset: number
  /** Length of record in data file */
  length: number
  /** First 8 bytes of key hash for validation */
  keyHash: Uint8Array
}

/**
 * Data record structure.
 * Variable size on disk, contains key and embedding.
 */
export interface DataRecord {
  /** Operation type (INSERT, UPDATE, DELETE) */
  opType: OpType
  /** Monotonic sequence number */
  sequenceNumber: bigint
  /** Unix timestamp in milliseconds when record was created */
  timestamp: bigint
  /** Record key */
  key: string
  /** Embedding dimension */
  dimension: number
  /** Embedding vector */
  embedding: Float32Array
}

/**
 * Options for configuring write batching.
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
 * Options for creating a storage engine.
 */
export interface StorageEngineOptions {
  /** Path to the data file (without extension) */
  dataPath: string
  /** Embedding dimension (default: 384 for bge-small-en-v1.5) */
  dimension?: number
  /** Lock acquisition timeout in milliseconds (default: 10000). Use 0 to fail immediately. */
  lockTimeout?: number
  /** Enable write batching for improved throughput (default: true) */
  batchingEnabled?: boolean
  /** Options for write batching */
  batchOptions?: WriteBatcherOptions
  /** Open database in read-only mode (default: false). Allows concurrent reads without exclusive lock. */
  readOnly?: boolean
}

/**
 * Result of deserializing a data record.
 */
export interface DeserializeDataResult {
  record: DataRecord
  bytesRead: number
}

/**
 * Result of deserializing a WAL entry.
 */
export interface DeserializeWalResult {
  entry: WalEntry
  bytesRead: number
}

/**
 * Header information for the data file.
 */
export interface DataFileHeader {
  /** Format version */
  version: number
  /** Embedding dimension */
  dimension: number
}
