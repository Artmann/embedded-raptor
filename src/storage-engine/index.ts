/**
 * Storage Engine module - WAL-based durable storage for embeddings.
 */

// Main classes
export { StorageEngine } from './storage-engine'
export { Wal } from './wal'
export { KeyIndex } from './key-index'
export { FileLock, DatabaseLockedError } from './file-lock'
export { Mutex } from './mutex'
export { WriteBatcher } from './write-batcher'

// Types
export type {
  OpType,
  RecordLocation,
  WalEntry,
  DataRecord,
  StorageEngineOptions,
  DeserializeDataResult,
  DeserializeWalResult,
  DataFileHeader,
  WriteBatcherOptions
} from './types'

export type { PendingWrite, FlushResult } from './write-batch'

// Constants
export {
  recordMagic,
  recordTrailer,
  headerMagic,
  headerVersionV1,
  headerVersionV2,
  headerSize,
  walEntrySize,
  opType,
  fileExtensions
} from './constants'

// Serialization utilities
export {
  serializeDataRecord,
  deserializeDataRecord,
  serializeHeader,
  deserializeHeader,
  readKeyFromBuffer,
  calculateRecordSize,
  crc32
} from './data-format'

export { serializeWalEntry, deserializeWalEntry, hashKey } from './wal-format'

// Migration
export { detectVersion, migrateV1ToV2, ensureV2Format } from './migration'
