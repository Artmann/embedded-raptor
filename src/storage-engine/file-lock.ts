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
        if (error instanceof Error && 'code' in error) {
          // EEXIST: Lock file exists - another process holds the lock
          if (error.code === 'EEXIST') {
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

          // EACCES/EROFS: Permission denied or read-only filesystem
          if (error.code === 'EACCES' || error.code === 'EROFS') {
            throw new LockPermissionError(this.filePath, error)
          }
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

/**
 * Error thrown when the lock file cannot be created due to permission issues.
 * This typically happens in read-only filesystems or restricted environments
 * like production containers.
 */
export class LockPermissionError extends Error {
  constructor(
    public readonly lockPath: string,
    public readonly originalError?: Error
  ) {
    super(
      `Permission denied when creating lock file: ${lockPath}\n\n` +
        `This usually happens in read-only filesystems (e.g., production containers).\n` +
        `Read operations do not require a lock, only write operations.\n` +
        `If you need to write, ensure the directory is writable.`
    )
    this.name = 'LockPermissionError'
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
