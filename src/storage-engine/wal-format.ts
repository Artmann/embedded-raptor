/**
 * WAL entry serialization and deserialization.
 *
 * Fixed 48-byte format:
 * [magic:4][version:2][opType:1][flags:1][seqNum:8][offset:8][length:4][keyHash:8][reserved:4][checksum:4][trailer:4]
 */

import {
  recordMagic,
  recordTrailer,
  walVersion,
  walEntrySize,
  walEntryLayout
} from './constants'
import { crc32 } from './data-format'
import type { WalEntry, DeserializeWalResult, OpType } from './types'

/**
 * Compute an 8-byte hash of a key for validation during recovery.
 * Uses FNV-1a hash truncated to 8 bytes.
 */
export function hashKey(key: string): Uint8Array {
  const keyBytes = new TextEncoder().encode(key)
  let hash = BigInt('0xcbf29ce484222325') // FNV offset basis

  for (let i = 0; i < keyBytes.length; i++) {
    hash ^= BigInt(keyBytes[i])
    hash = BigInt.asUintN(64, hash * BigInt('0x100000001b3')) // FNV prime
  }

  const result = new Uint8Array(8)
  const view = new DataView(result.buffer)
  view.setBigUint64(0, hash, true)
  return result
}

/**
 * Serialize a WAL entry to a fixed 48-byte buffer.
 */
export function serializeWalEntry(entry: WalEntry): Uint8Array {
  const buffer = new Uint8Array(walEntrySize)
  const view = new DataView(buffer.buffer)

  // Magic (4 bytes)
  view.setUint32(walEntryLayout.magic, recordMagic, true)

  // Version (2 bytes)
  view.setUint16(walEntryLayout.version, walVersion, true)

  // OpType (1 byte)
  view.setUint8(walEntryLayout.opType, entry.opType)

  // Flags (1 byte) - reserved
  view.setUint8(walEntryLayout.flags, 0)

  // Sequence number (8 bytes)
  view.setBigInt64(walEntryLayout.seqNum, entry.sequenceNumber, true)

  // Offset (8 bytes)
  view.setBigUint64(walEntryLayout.offset, BigInt(entry.offset), true)

  // Length (4 bytes)
  view.setUint32(walEntryLayout.length, entry.length, true)

  // Key hash (8 bytes)
  buffer.set(entry.keyHash.subarray(0, 8), walEntryLayout.keyHash)

  // Reserved (4 bytes) - already zero

  // Calculate checksum of everything before the checksum field
  const checksumData = buffer.subarray(0, walEntryLayout.checksum)
  const checksum = crc32(checksumData)
  view.setUint32(walEntryLayout.checksum, checksum, true)

  // Trailer (4 bytes)
  view.setUint32(walEntryLayout.trailer, recordTrailer, true)

  return buffer
}

/**
 * Deserialize a WAL entry from bytes.
 * Returns null if the entry is invalid or corrupted.
 */
export function deserializeWalEntry(
  data: Uint8Array,
  startOffset = 0
): DeserializeWalResult | null {
  // Check if we have enough data
  if (data.length - startOffset < walEntrySize) {
    return null
  }

  const view = new DataView(data.buffer, data.byteOffset + startOffset)

  // Validate magic
  const magic = view.getUint32(walEntryLayout.magic, true)
  if (magic !== recordMagic) {
    return null
  }

  // Validate version
  const version = view.getUint16(walEntryLayout.version, true)
  if (version !== walVersion) {
    return null
  }

  // Read fields
  const opType = view.getUint8(walEntryLayout.opType) as OpType
  const sequenceNumber = view.getBigInt64(walEntryLayout.seqNum, true)
  const offset = Number(view.getBigUint64(walEntryLayout.offset, true))
  const length = view.getUint32(walEntryLayout.length, true)
  const keyHash = data.slice(
    startOffset + walEntryLayout.keyHash,
    startOffset + walEntryLayout.keyHash + 8
  )

  // Validate checksum
  const storedChecksum = view.getUint32(walEntryLayout.checksum, true)
  const checksumData = data.subarray(
    startOffset,
    startOffset + walEntryLayout.checksum
  )
  const computedChecksum = crc32(checksumData)
  if (storedChecksum !== computedChecksum) {
    return null
  }

  // Validate trailer
  const trailer = view.getUint32(walEntryLayout.trailer, true)
  if (trailer !== recordTrailer) {
    return null
  }

  return {
    entry: {
      opType,
      sequenceNumber,
      offset,
      length,
      keyHash
    },
    bytesRead: walEntrySize
  }
}
