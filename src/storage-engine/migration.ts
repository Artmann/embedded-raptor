/**
 * Migration utilities for converting v1 (old binary format) to v2 (WAL format).
 *
 * v1 format:
 * - Header: "EMBD" (4) + version (2) + dimension (4) + reserved (6) = 16 bytes
 * - Records: keyLen (2) + key (N) + embedding (D*4) + recordLenFooter (4)
 *
 * v2 format:
 * - Uses WAL-based storage with checksums and sequence numbers
 */

import { open, stat, copyFile, rm } from 'node:fs/promises'
import {
  headerMagic,
  headerVersionV1,
  headerVersionV2,
  headerSize
} from './constants'
// Data format functions not used directly but may be needed for future enhancements
import { StorageEngine } from './storage-engine'

/**
 * Detect the format version of a database file.
 * Returns null if the file doesn't exist or is not a valid database.
 */
export async function detectVersion(filePath: string): Promise<number | null> {
  try {
    const fileHandle = await open(filePath, 'r')
    try {
      const buffer = new Uint8Array(headerSize)
      await fileHandle.read(buffer, 0, headerSize, 0)

      const view = new DataView(buffer.buffer)
      const magic = view.getUint32(0, true)

      if (magic !== headerMagic) {
        return null
      }

      return view.getUint16(4, true)
    } finally {
      await fileHandle.close()
    }
  } catch {
    return null
  }
}

/**
 * Read all entries from a v1 format file.
 */
async function* readV1Entries(
  filePath: string
): AsyncGenerator<{ key: string; embedding: Float32Array }> {
  const fileHandle = await open(filePath, 'r')
  try {
    // Read header
    const headerBuffer = new Uint8Array(headerSize)
    await fileHandle.read(headerBuffer, 0, headerSize, 0)

    const view = new DataView(headerBuffer.buffer)
    const dimension = view.getUint32(6, true)

    // Get file size
    const stats = await stat(filePath)
    const fileSize = stats.size

    if (fileSize <= headerSize) {
      return
    }

    // Track seen keys for deduplication (we read forward, so last one wins)
    // But v1 reader reads backward for dedup - we'll keep all and let caller handle
    const seenKeys = new Map<string, { key: string; embedding: Float32Array }>()

    // Read records forward
    let offset = headerSize
    while (offset < fileSize) {
      // Read key length
      const keyLenBuffer = new Uint8Array(2)
      const { bytesRead: keyLenBytes } = await fileHandle.read(
        keyLenBuffer,
        0,
        2,
        offset
      )
      if (keyLenBytes < 2) {
        break
      }

      const keyLen = new DataView(keyLenBuffer.buffer).getUint16(0, true)
      const recordLength = 2 + keyLen + dimension * 4 + 4

      // Read full record
      const recordBuffer = new Uint8Array(recordLength)
      const { bytesRead } = await fileHandle.read(
        recordBuffer,
        0,
        recordLength,
        offset
      )
      if (bytesRead < recordLength) {
        break
      }

      // Parse key
      const keyBytes = recordBuffer.slice(2, 2 + keyLen)
      const key = new TextDecoder().decode(keyBytes)

      // Parse embedding
      const embeddingView = new DataView(
        recordBuffer.buffer,
        2 + keyLen,
        dimension * 4
      )
      const embedding = new Float32Array(dimension)
      for (let i = 0; i < dimension; i++) {
        embedding[i] = embeddingView.getFloat32(i * 4, true)
      }

      // Store (newer overwrites older)
      seenKeys.set(key, { key, embedding })

      offset += recordLength
    }

    // Yield all entries
    for (const entry of seenKeys.values()) {
      yield entry
    }
  } finally {
    await fileHandle.close()
  }
}

/**
 * Migrate a v1 database to v2 format.
 * Creates a backup of the original file before migration.
 */
export async function migrateV1ToV2(
  sourcePath: string,
  backupPath?: string
): Promise<void> {
  // Detect version
  const version = await detectVersion(sourcePath)
  if (version === null) {
    throw new Error(`Invalid database file: ${sourcePath}`)
  }
  if (version === headerVersionV2) {
    // Already v2, nothing to do
    return
  }
  if (version !== headerVersionV1) {
    throw new Error(`Unknown database version: ${version}`)
  }

  // Create backup
  const backup = backupPath ?? `${sourcePath}.v1.backup`
  await copyFile(sourcePath, backup)

  // Get dimension from v1 file
  const fileHandle = await open(sourcePath, 'r')
  let dimension: number
  try {
    const headerBuffer = new Uint8Array(headerSize)
    await fileHandle.read(headerBuffer, 0, headerSize, 0)
    const view = new DataView(headerBuffer.buffer)
    dimension = view.getUint32(6, true)
  } finally {
    await fileHandle.close()
  }

  // Read all v1 entries
  const entries: Array<{ key: string; embedding: Float32Array }> = []
  for await (const entry of readV1Entries(sourcePath)) {
    entries.push(entry)
  }

  // Remove old files
  const basePath = sourcePath.replace(/\.[^.]+$/, '')
  await rm(sourcePath, { force: true })
  await rm(`${basePath}.raptor-wal`, { force: true })
  await rm(`${basePath}.raptor.lock`, { force: true })

  // Create new v2 storage engine and write all entries
  const engine = await StorageEngine.create({
    dataPath: basePath,
    dimension
  })

  try {
    for (const entry of entries) {
      await engine.writeRecord(entry.key, entry.embedding)
    }
  } finally {
    await engine.close()
  }

  console.log(`Migrated ${entries.length} entries from v1 to v2 format`)
  console.log(`Backup saved to: ${backup}`)
}

/**
 * Check if migration is needed and perform it automatically.
 * Returns the path to use (may be different if migration occurred).
 */
export async function ensureV2Format(
  dataPath: string,
  _dimension: number
): Promise<void> {
  const version = await detectVersion(dataPath)

  if (version === null) {
    // Fresh database, nothing to do
    return
  }

  if (version === headerVersionV1) {
    console.log(
      `Detected v1 format database at ${dataPath}, migrating to v2...`
    )
    await migrateV1ToV2(dataPath)
    console.log('Migration complete.')
  } else if (version !== headerVersionV2) {
    throw new Error(`Unknown database version: ${version}`)
  }
}
