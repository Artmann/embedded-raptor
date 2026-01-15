/**
 * In-memory key index for O(1) lookups.
 *
 * Maps keys to their location in the data file. This index is
 * rebuilt from the WAL on startup and updated after each write.
 */

import { open } from 'node:fs/promises'
import { opType } from './constants'
import { readKeyFromBuffer } from './data-format'
import type { Wal } from './wal'
import type { RecordLocation, WalEntry, OpType } from './types'

export interface BuildFromWalResult {
  index: KeyIndex
  maxSequence: bigint
}

export class KeyIndex {
  private readonly entries: Map<string, RecordLocation>

  private constructor() {
    this.entries = new Map()
  }

  /**
   * Create an empty index.
   */
  static create(): KeyIndex {
    return new KeyIndex()
  }

  /**
   * Build the index from a WAL by replaying all entries.
   * Also reads keys from the data file since WAL only stores hashes.
   * Returns both the index and the maximum sequence number seen.
   */
  static async buildFromWal(
    wal: Wal,
    dataPath: string
  ): Promise<BuildFromWalResult> {
    const index = new KeyIndex()
    let maxSequence = 0n

    // Open data file for reading keys
    let dataHandle
    try {
      dataHandle = await open(dataPath, 'r')
    } catch {
      // No data file = fresh database
      return { index, maxSequence }
    }

    try {
      // Replay WAL entries
      for await (const entry of wal.recover()) {
        // Track max sequence number
        if (entry.sequenceNumber > maxSequence) {
          maxSequence = entry.sequenceNumber
        }

        // Read key from data file at the offset
        // We write full data records for all operations including deletes
        const keyBuffer = new Uint8Array(1024) // Max key size
        await dataHandle.read(keyBuffer, 0, 1024, entry.offset)
        const key = readKeyFromBuffer(keyBuffer)

        if (key) {
          index.applyEntry(key, entry)
        }
      }
    } finally {
      await dataHandle.close()
    }

    return { index, maxSequence }
  }

  /**
   * Apply a WAL entry to update the index.
   */
  applyEntry(key: string, entry: WalEntry): void {
    if (entry.opType === opType.delete) {
      this.entries.delete(key)
    } else {
      this.entries.set(key, {
        offset: entry.offset,
        length: entry.length,
        sequenceNumber: entry.sequenceNumber
      })
    }
  }

  /**
   * Apply a location update directly (used after writes).
   */
  apply(key: string, location: RecordLocation, op: OpType): void {
    if (op === opType.delete) {
      this.entries.delete(key)
    } else {
      this.entries.set(key, location)
    }
  }

  /**
   * Get the location of a key.
   */
  get(key: string): RecordLocation | undefined {
    return this.entries.get(key)
  }

  /**
   * Check if a key exists.
   */
  has(key: string): boolean {
    return this.entries.has(key)
  }

  /**
   * Delete a key from the index.
   */
  delete(key: string): boolean {
    return this.entries.delete(key)
  }

  /**
   * Get all keys in the index.
   */
  keys(): IterableIterator<string> {
    return this.entries.keys()
  }

  /**
   * Iterate over all entries as [key, location] pairs.
   */
  locations(): IterableIterator<[string, RecordLocation]> {
    return this.entries.entries()
  }

  /**
   * Get the number of entries in the index.
   */
  count(): number {
    return this.entries.size
  }
}
