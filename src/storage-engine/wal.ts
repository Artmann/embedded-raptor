/**
 * Write-Ahead Log (WAL) implementation.
 *
 * The WAL provides durability by recording all operations before
 * they are applied to the main data file. On recovery, the WAL
 * is scanned to rebuild the in-memory index.
 */

import { open, stat, mkdir } from 'node:fs/promises'
import type { FileHandle } from 'node:fs/promises'
import { dirname } from 'node:path'
import { walEntrySize } from './constants'
import { serializeWalEntry, deserializeWalEntry } from './wal-format'
import type { WalEntry } from './types'

export class Wal {
  private readonly filePath: string
  private fileHandle: FileHandle | null = null

  constructor(filePath: string) {
    this.filePath = filePath
  }

  /**
   * Append a WAL entry and sync to disk.
   * This is the commit point - once this returns, the operation is durable.
   */
  async append(entry: WalEntry): Promise<void> {
    const buffer = serializeWalEntry(entry)

    // Ensure directory exists
    await mkdir(dirname(this.filePath), { recursive: true })

    // Open file for appending if not already open
    this.fileHandle ??= await open(this.filePath, 'a')

    // Write and sync
    await this.fileHandle.write(buffer)
    await this.fileHandle.sync()
  }

  /**
   * Recover WAL entries from disk.
   * Yields valid entries in order, stopping at the first corrupted entry.
   */
  async *recover(): AsyncGenerator<WalEntry> {
    // Check if WAL file exists
    let fileStats
    try {
      fileStats = await stat(this.filePath)
    } catch {
      // No WAL file = fresh database
      return
    }

    if (fileStats.size === 0) {
      return
    }

    // Read entire WAL into memory (it's fixed-size entries, relatively small)
    const fileHandle = await open(this.filePath, 'r')
    try {
      const buffer = new Uint8Array(fileStats.size)
      await fileHandle.read(buffer, 0, fileStats.size, 0)

      let offset = 0
      while (offset + walEntrySize <= buffer.length) {
        const result = deserializeWalEntry(buffer, offset)

        if (!result) {
          // Corrupted entry - stop recovery here
          // This is safe: we only process valid entries
          break
        }

        yield result.entry
        offset += walEntrySize
      }
    } finally {
      await fileHandle.close()
    }
  }

  /**
   * Close the WAL file handle.
   */
  async close(): Promise<void> {
    if (this.fileHandle) {
      await this.fileHandle.close()
      this.fileHandle = null
    }
  }

  /**
   * Get the file path for this WAL.
   */
  getFilePath(): string {
    return this.filePath
  }
}
