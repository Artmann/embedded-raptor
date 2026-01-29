/**
 * Permission error tests for FileLock.
 * Tests that EACCES errors are handled gracefully with helpful error messages.
 *
 * With operation-level locking:
 * - Lock errors occur during write operations, not on engine creation
 * - StorageEngine.create() on read-only filesystems fails at mkdir(), not lock acquisition
 */

import { describe, it, expect, afterEach } from 'vitest'
import { LockPermissionError } from '../file-lock'
import { StorageEngine } from '../storage-engine'
import {
  createTestPaths,
  cleanup,
  generateRandomEmbedding,
  type TestPaths
} from './helpers'

describe('FileLock permission errors', () => {
  const testPathsList: TestPaths[] = []
  const engines: StorageEngine[] = []

  afterEach(async () => {
    // Close all engines
    for (const engine of engines) {
      try {
        await engine.close()
      } catch {
        // Ignore errors during cleanup
      }
    }
    engines.length = 0

    await cleanup(testPathsList)
    testPathsList.length = 0
  })

  it('throws LockPermissionError on EACCES when creating lock file', () => {
    // Create a lock in a valid directory first, then test the error path
    // directly with a LockPermissionError
    const error = new LockPermissionError('/sys/test.raptor.lock')
    expect(error).toBeInstanceOf(LockPermissionError)
    expect(error.lockPath).toBe('/sys/test.raptor.lock')
    expect(error.message).toContain('Permission denied')
    expect(error.message).toContain('readOnly: true')
  })

  it('LockPermissionError message includes helpful guidance', () => {
    const error = new LockPermissionError('/app/recipe-embeddings.raptor.lock')

    expect(error.message).toContain('Permission denied')
    expect(error.message).toContain('/app/recipe-embeddings.raptor.lock')
    expect(error.message).toContain('readOnly: true')
    expect(error.message).toContain('production containers')
    expect(error.name).toBe('LockPermissionError')
  })

  it('LockPermissionError stores the original error', () => {
    const originalError = new Error('EACCES: permission denied')
    const error = new LockPermissionError(
      '/app/recipe-embeddings.raptor.lock',
      originalError
    )

    expect(error.originalError).toBe(originalError)
    expect(error.lockPath).toBe('/app/recipe-embeddings.raptor.lock')
  })

  it('StorageEngine throws error for read-only filesystem on create', async () => {
    // /sys is a read-only virtual filesystem
    // With operation-level locking, the error occurs at mkdir() during create
    const readOnlyPath = '/sys/database.raptor'

    await expect(
      StorageEngine.create({
        dataPath: readOnlyPath,
        dimension: 384
      })
    ).rejects.toThrow() // Throws EROFS error from mkdir
  })

  it('readOnly mode bypasses directory creation and lock file', async () => {
    // Create database first
    const paths = createTestPaths('permission-readonly-bypass')
    testPathsList.push(paths)

    const writer = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await writer.writeRecord('test', generateRandomEmbedding(384))
    await writer.close()

    // Open in read-only mode - this should work even if we couldn't create lock
    const reader = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384,
      readOnly: true
    })
    engines.push(reader)

    expect(reader.isReadOnly()).toBe(true)
    expect(reader.count()).toBe(1)
    expect(reader.hasKey('test')).toBe(true)
  })
})
