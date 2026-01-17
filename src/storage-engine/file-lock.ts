import { mkdir, open, rm } from 'node:fs/promises'
import { dirname } from 'node:path'

const defaultLockTimeout = 10_000
const retryInterval = 100

/**
 * Exclusive file lock for preventing multiple processes from
 * accessing the same database simultaneously.
 *
 * Uses atomic file creation with O_EXCL flag to ensure only one
 * process can acquire the lock at a time.
 */
export class FileLock {
  private locked = false
  private readonly filePath: string
  private readonly timeoutMs: number

  constructor(filePath: string, timeoutMs: number = defaultLockTimeout) {
    this.filePath = filePath
    this.timeoutMs = timeoutMs
  }

  /**
   * Acquire an exclusive lock on the file.
   * Creates the lock file if it doesn't exist.
   * Retries for up to timeoutMs before throwing DatabaseLockedError.
   */
  async acquire(): Promise<void> {
    if (this.locked) {
      throw new Error('Lock already acquired')
    }

    // Ensure parent directory exists
    await mkdir(dirname(this.filePath), { recursive: true })

    const startTime = Date.now()

    // eslint-disable-next-line no-constant-condition
    while (true) {
      try {
        // Try to create lock file exclusively (O_CREAT | O_EXCL | O_WRONLY)
        // This is atomic - only one process can succeed
        // Using 'wx' string flag for Bun compatibility on Windows
        const fileHandle = await open(this.filePath, 'wx')

        // Write our PID to the lock file for debugging
        await fileHandle.write(`${process.pid}\n`)
        await fileHandle.close()

        this.locked = true
        return
      } catch (error) {
        if (
          error instanceof Error &&
          'code' in error &&
          error.code === 'EEXIST'
        ) {
          // Lock file exists - check if we should retry
          const elapsed = Date.now() - startTime
          if (elapsed >= this.timeoutMs) {
            throw new DatabaseLockedError(
              `Database is locked by another process (timeout after ${this.timeoutMs}ms): ${this.filePath}`
            )
          }

          // Wait before retrying
          await sleep(retryInterval)
          continue
        }
        throw error
      }
    }
  }

  /**
   * Release the lock by deleting the lock file.
   */
  async release(): Promise<void> {
    if (!this.locked) {
      return
    }

    try {
      await rm(this.filePath, { force: true })
    } finally {
      this.locked = false
    }
  }

  /**
   * Check if this instance currently holds the lock.
   */
  isLocked(): boolean {
    return this.locked
  }
}

/**
 * Error thrown when attempting to open a database that is
 * already locked by another process.
 */
export class DatabaseLockedError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'DatabaseLockedError'
  }
}

/**
 * Error thrown when attempting to write to a database opened in read-only mode.
 */
export class ReadOnlyError extends Error {
  constructor(message: string = 'Cannot write to a read-only database') {
    super(message)
    this.name = 'ReadOnlyError'
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
