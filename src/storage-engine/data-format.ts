/**
 * Data record serialization and deserialization.
 *
 * Binary format (v2):
 * [magic:4][version:2][opType:1][flags:1][seqNum:8][timestamp:8][keyLen:2][key:N][dimension:4][embedding:D*4][checksum:4][trailer:4]
 */

import {
  recordMagic,
  recordTrailer,
  recordVersion,
  headerMagic,
  headerVersionV2,
  headerSize,
  headerOffsets
} from './constants'
import type {
  DataRecord,
  DeserializeDataResult,
  DataFileHeader,
  OpType
} from './types'

/**
 * Calculate the size of a serialized data record.
 */
export function calculateRecordSize(
  keyLength: number,
  dimension: number
): number {
  // magic(4) + version(2) + opType(1) + flags(1) + seqNum(8) + timestamp(8) + keyLen(2) + key(N) + dimension(4) + embedding(D*4) + checksum(4) + trailer(4)
  return 4 + 2 + 1 + 1 + 8 + 8 + 2 + keyLength + 4 + dimension * 4 + 4 + 4
}

/**
 * Serialize a data record to bytes.
 */
export function serializeDataRecord(record: DataRecord): Uint8Array {
  const keyBytes = new TextEncoder().encode(record.key)
  const size = calculateRecordSize(keyBytes.length, record.dimension)
  const buffer = new Uint8Array(size)
  const view = new DataView(buffer.buffer)

  let offset = 0

  // Magic (4 bytes)
  view.setUint32(offset, recordMagic, true)
  offset += 4

  // Version (2 bytes)
  view.setUint16(offset, recordVersion, true)
  offset += 2

  // OpType (1 byte)
  view.setUint8(offset, record.opType)
  offset += 1

  // Flags (1 byte) - reserved for future use
  view.setUint8(offset, 0)
  offset += 1

  // Sequence number (8 bytes)
  view.setBigInt64(offset, record.sequenceNumber, true)
  offset += 8

  // Timestamp (8 bytes) - Unix milliseconds
  view.setBigInt64(offset, record.timestamp, true)
  offset += 8

  // Key length (2 bytes)
  view.setUint16(offset, keyBytes.length, true)
  offset += 2

  // Key (variable)
  buffer.set(keyBytes, offset)
  offset += keyBytes.length

  // Dimension (4 bytes)
  view.setUint32(offset, record.dimension, true)
  offset += 4

  // Embedding (dimension * 4 bytes)
  for (let i = 0; i < record.dimension; i++) {
    view.setFloat32(offset, record.embedding[i], true)
    offset += 4
  }

  // Calculate CRC32 checksum of everything before the checksum field
  const checksumData = buffer.subarray(0, offset)
  const checksum = crc32(checksumData)
  view.setUint32(offset, checksum, true)
  offset += 4

  // Trailer (4 bytes)
  view.setUint32(offset, recordTrailer, true)

  return buffer
}

/**
 * Deserialize a data record from bytes.
 * Returns null if the record is invalid or corrupted.
 */
export function deserializeDataRecord(
  data: Uint8Array,
  startOffset = 0
): DeserializeDataResult | null {
  const view = new DataView(data.buffer, data.byteOffset + startOffset)

  // Minimum size check: header fields (26 bytes) + minimal key + minimal embedding + checksum + trailer
  if (data.length - startOffset < 34) {
    return null
  }

  let offset = 0

  // Validate magic
  const magic = view.getUint32(offset, true)
  if (magic !== recordMagic) {
    return null
  }
  offset += 4

  // Version
  const version = view.getUint16(offset, true)
  if (version !== recordVersion) {
    return null
  }
  offset += 2

  // OpType
  const opType = view.getUint8(offset) as OpType
  offset += 1

  // Flags (skip)
  offset += 1

  // Sequence number
  const sequenceNumber = view.getBigInt64(offset, true)
  offset += 8

  // Timestamp
  const timestamp = view.getBigInt64(offset, true)
  offset += 8

  // Key length
  const keyLen = view.getUint16(offset, true)
  offset += 2

  // Check if we have enough data for key
  if (data.length - startOffset < offset + keyLen + 12) {
    return null
  }

  // Key
  const keyBytes = data.subarray(
    startOffset + offset,
    startOffset + offset + keyLen
  )
  const key = new TextDecoder().decode(keyBytes)
  offset += keyLen

  // Dimension
  const dimension = view.getUint32(offset, true)
  offset += 4

  // Check if we have enough data for embedding + checksum + trailer
  const embeddingSize = dimension * 4
  if (data.length - startOffset < offset + embeddingSize + 8) {
    return null
  }

  // Embedding
  const embedding = new Float32Array(dimension)
  for (let i = 0; i < dimension; i++) {
    embedding[i] = view.getFloat32(offset, true)
    offset += 4
  }

  // Checksum
  const storedChecksum = view.getUint32(offset, true)
  const checksumData = data.subarray(startOffset, startOffset + offset)
  const computedChecksum = crc32(checksumData)
  if (storedChecksum !== computedChecksum) {
    return null
  }
  offset += 4

  // Trailer
  const trailer = view.getUint32(offset, true)
  if (trailer !== recordTrailer) {
    return null
  }
  offset += 4

  return {
    record: {
      opType,
      sequenceNumber,
      timestamp,
      key,
      dimension,
      embedding
    },
    bytesRead: offset
  }
}

/**
 * Read just the key from a data record at a given offset.
 * Useful during WAL recovery to avoid reading the entire embedding.
 */
export function readKeyFromBuffer(
  data: Uint8Array,
  startOffset = 0
): string | null {
  const view = new DataView(data.buffer, data.byteOffset + startOffset)

  // Check minimum size for header + keyLen (26 bytes: magic(4) + version(2) + opType(1) + flags(1) + seqNum(8) + timestamp(8) + keyLen(2))
  if (data.length - startOffset < 26) {
    return null
  }

  // Validate magic
  const magic = view.getUint32(0, true)
  if (magic !== recordMagic) {
    return null
  }

  // Key length is at offset 24 (after timestamp)
  const keyLen = view.getUint16(24, true)

  // Check if we have enough data for the key
  if (data.length - startOffset < 26 + keyLen) {
    return null
  }

  const keyBytes = data.subarray(startOffset + 26, startOffset + 26 + keyLen)
  return new TextDecoder().decode(keyBytes)
}

/**
 * Serialize the file header.
 */
export function serializeHeader(dimension: number): Uint8Array {
  const buffer = new Uint8Array(headerSize)
  const view = new DataView(buffer.buffer)

  // Magic (4 bytes)
  view.setUint32(headerOffsets.magic, headerMagic, true)

  // Version (2 bytes)
  view.setUint16(headerOffsets.version, headerVersionV2, true)

  // Dimension (4 bytes)
  view.setUint32(headerOffsets.dimension, dimension, true)

  // Reserved bytes are already zero

  return buffer
}

/**
 * Deserialize the file header.
 */
export function deserializeHeader(data: Uint8Array): DataFileHeader | null {
  if (data.length < headerSize) {
    return null
  }

  const view = new DataView(data.buffer, data.byteOffset)

  // Validate magic
  const magic = view.getUint32(headerOffsets.magic, true)
  if (magic !== headerMagic) {
    return null
  }

  const version = view.getUint16(headerOffsets.version, true)
  const dimension = view.getUint32(headerOffsets.dimension, true)

  return { version, dimension }
}

/**
 * CRC32 implementation using the standard polynomial.
 * This matches Bun.hash.crc32() behavior.
 */
const crc32Table = makeCrc32Table()

function makeCrc32Table(): Uint32Array {
  const table = new Uint32Array(256)
  for (let i = 0; i < 256; i++) {
    let c = i
    for (let j = 0; j < 8; j++) {
      if (c & 1) {
        c = 0xedb88320 ^ (c >>> 1)
      } else {
        c = c >>> 1
      }
    }
    table[i] = c
  }
  return table
}

export function crc32(data: Uint8Array): number {
  let crc = 0xffffffff
  for (let i = 0; i < data.length; i++) {
    crc = crc32Table[(crc ^ data[i]) & 0xff] ^ (crc >>> 8)
  }
  return (crc ^ 0xffffffff) >>> 0
}
