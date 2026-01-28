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
    }
  })

  it('LockPermissionError message includes helpful guidance', () => {
    const error = new LockPermissionError('/app/recipe-embeddings.raptor.lock')

    expect(error.message).toContain('Permission denied')
    expect(error.message).toContain('/app/recipe-embeddings.raptor.lock')
    expect(error.message).toContain('read-only filesystems')
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

  it('StorageEngine throws LockPermissionError on write to read-only filesystem', async () => {
    // /sys is a read-only virtual filesystem
    const readOnlyPath = '/sys/database.raptor'

    // Creating the engine should work (no lock acquired yet)
    const engine = await StorageEngine.create({
      dataPath: readOnlyPath,
      dimension: 384
    })

    // But writing should fail with permission error
    await expect(
      engine.writeRecord('test', generateRandomEmbedding(384))
    ).rejects.toThrow(LockPermissionError)
  })

  it('read operations do not require lock file creation', async () => {
    // Create database first
    const paths = createTestPaths('permission-no-lock-for-read')
    testPathsList.push(paths)

    const writer = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await writer.writeRecord('test', generateRandomEmbedding(384))
    await writer.close()

    // Open another engine - no lock needed for reads
    const reader = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(reader)

    // Should not have lock (only reads performed)
    expect(reader.hasWriteLock()).toBe(false)
    expect(reader.count()).toBe(1)
    expect(reader.hasKey('test')).toBe(true)

    // Can read the record
    const record = await reader.readRecord('test')
    expect(record).not.toBeNull()

    // Still no lock acquired
    expect(reader.hasWriteLock()).toBe(false)
  })
})
