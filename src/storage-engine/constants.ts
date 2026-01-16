/**
 * Constants for the WAL-based storage engine.
 *
 * These define the binary format for data records and WAL entries.
 */

// Magic numbers for format validation
export const recordMagic = 0xcafebabe
export const recordTrailer = 0xdeadbeef

// File header magic (ASCII "EMBD")
export const headerMagic = 0x454d4244

// Format versions
export const headerVersionV1 = 1 // Old binary format (no WAL)
export const headerVersionV2 = 2 // WAL-enabled format
export const recordVersion = 2 // v2 includes timestamp field
export const walVersion = 1

// Fixed sizes
export const headerSize = 16
export const walEntrySize = 48

// Data record field offsets
export const dataRecordOffsets = {
  magic: 0, // 4 bytes
  version: 4, // 2 bytes
  opType: 6, // 1 byte
  flags: 7, // 1 byte
  seqNum: 8, // 8 bytes (BigInt64)
  timestamp: 16, // 8 bytes (BigInt64) - Unix milliseconds
  keyLen: 24, // 2 bytes
  key: 26 // variable, followed by dimension (4), embedding (D*4), checksum (4), trailer (4)
} as const

// WAL entry layout (48 bytes total)
export const walEntryLayout = {
  magic: 0, // 4 bytes
  version: 4, // 2 bytes
  opType: 6, // 1 byte
  flags: 7, // 1 byte
  seqNum: 8, // 8 bytes
  offset: 16, // 8 bytes
  length: 24, // 4 bytes
  keyHash: 28, // 8 bytes (truncated hash for validation)
  reserved: 36, // 4 bytes (padding for future use)
  checksum: 40, // 4 bytes
  trailer: 44 // 4 bytes
} as const

// Header field offsets
export const headerOffsets = {
  magic: 0, // 4 bytes
  version: 4, // 2 bytes
  dimension: 6, // 4 bytes
  reserved: 10 // 6 bytes
} as const

// Operation types
export const opType = {
  insert: 0,
  update: 1,
  delete: 2
} as const

// Default file extensions
export const fileExtensions = {
  data: '.raptor',
  wal: '.raptor-wal',
  lock: '.raptor.lock'
} as const
