/**
 * Permission error tests for FileLock.
 * Tests that EACCES errors are handled gracefully with helpful error messages.
 */

import { describe, it, expect, afterEach } from 'vitest'
import { FileLock, LockPermissionError } from '../file-lock'
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

  it('throws LockPermissionError on EACCES when creating lock file in /sys', async () => {
    // /sys is a read-only virtual filesystem - we can't create files there
    const lockPath = '/sys/test-lock-file.raptor.lock'

    const lock = new FileLock(lockPath)
    await expect(lock.acquire()).rejects.toThrow(LockPermissionError)

    try {
      await lock.acquire()
    } catch (error) {
      expect(error).toBeInstanceOf(LockPermissionError)
      const permError = error as LockPermissionError
      expect(permError.lockPath).toBe(lockPath)
      expect(permError.message).toContain('Permission denied')
      expect(permError.message).toContain('readOnly: true')
    }
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

  it('StorageEngine throws LockPermissionError for read-only filesystem', async () => {
    // /sys is a read-only virtual filesystem
    const readOnlyPath = '/sys/database.raptor'

    await expect(
      StorageEngine.create({
        dataPath: readOnlyPath,
        dimension: 384
      })
    ).rejects.toThrow(LockPermissionError)
  })

  it('readOnly mode bypasses lock file creation entirely', async () => {
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
