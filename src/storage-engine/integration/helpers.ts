import { unlink, rm } from 'node:fs/promises'
import { walEntrySize } from '../constants'

export { walEntrySize }

export interface TestPaths {
  dataPath: string
  walPath: string
  lockPath: string
}

/**
 * Generate unique test file paths with a prefix.
 */
export function createTestPaths(prefix: string): TestPaths {
  const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`
  const basePath = `/tmp/test-${prefix}-${id}`
  const dataPath = `${basePath}.raptor`
  const walPath = `${basePath}.raptor-wal`
  const lockPath = `${basePath}.raptor.lock`
  return { dataPath, walPath, lockPath }
}

/**
 * Clean up test files.
 */
export async function cleanup(paths: TestPaths[]): Promise<void> {
  for (const { dataPath, walPath, lockPath } of paths) {
    try {
      await unlink(dataPath)
    } catch {
      // File may not exist
    }
    try {
      await unlink(walPath)
    } catch {
      // File may not exist
    }
    try {
      await rm(lockPath, { force: true })
    } catch {
      // File may not exist
    }
  }
}

/**
 * Flip a single byte in a byte array at the specified index.
 */
export function flipByte(data: Uint8Array, index: number): void {
  if (index >= 0 && index < data.length) {
    data[index] = data[index] ^ 0xff
  }
}

/**
 * Collect all entries from an async generator into an array.
 */
export async function collectEntries<T>(
  generator: AsyncGenerator<T>
): Promise<T[]> {
  const entries: T[] = []
  for await (const entry of generator) {
    entries.push(entry)
  }
  return entries
}

/**
 * Generate a random embedding vector of the given dimension.
 */
export function generateRandomEmbedding(dimension: number = 384): Float32Array {
  const embedding = new Float32Array(dimension)
  for (let i = 0; i < dimension; i++) {
    embedding[i] = Math.random() * 2 - 1 // Values between -1 and 1
  }
  return embedding
}

/**
 * Check if two embeddings are approximately equal.
 */
export function embeddingsEqual(
  a: Float32Array | number[],
  b: Float32Array | number[],
  tolerance: number = 1e-6
): boolean {
  if (a.length !== b.length) {
    return false
  }
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > tolerance) {
      return false
    }
  }
  return true
}
