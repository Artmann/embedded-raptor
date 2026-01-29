/* eslint-disable @typescript-eslint/no-non-null-assertion */
/**
 * Database locking integration tests for StorageEngine.
 * Tests operation-level file locking behavior.
 *
 * With operation-level locking:
 * - Engine creation does NOT acquire any lock
 * - Locks are acquired per write operation and released immediately after
 * - Multiple engine instances can coexist (single-process only)
 */

import { describe, it, expect, afterEach } from 'vitest'
import { stat } from 'node:fs/promises'
import { StorageEngine } from '../storage-engine'
import { fileExtensions } from '../constants'
import {
  createTestPaths,
  cleanup,
  generateRandomEmbedding,
  embeddingsEqual,
  type TestPaths
} from './helpers'

describe('StorageEngine locking', () => {
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

  it('storage engine opens successfully and can write', async () => {
    const paths = createTestPaths('locking-open')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(engine)

    await engine.writeRecord('test', generateRandomEmbedding(384))

    expect(engine.count()).toBe(1)
  })

  it('engine creation does not create lock file', async () => {
    const paths = createTestPaths('no-lock-on-create')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(engine)

    // Lock file should not exist after creating engine (before any writes)
    const basePath = paths.dataPath.replace(/\.[^.]+$/, '')
    const lockPath = basePath + fileExtensions.lock
    const lockExists = await stat(lockPath).catch(() => null)

    expect(lockExists).toBeNull()
  })

  it('lock file is cleaned up after write operation', async () => {
    const paths = createTestPaths('lock-cleanup')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(engine)

    await engine.writeRecord('test', generateRandomEmbedding(384))

    // Lock file should be removed after write completes
    const basePath = paths.dataPath.replace(/\.[^.]+$/, '')
    const lockPath = basePath + fileExtensions.lock
    const lockExists = await stat(lockPath).catch(() => null)

    expect(lockExists).toBeNull()
  })

  it('multiple engine instances can coexist in same process', async () => {
    const paths = createTestPaths('multi-instance')
    testPathsList.push(paths)

    const engine1 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(engine1)

    await engine1.writeRecord('first', generateRandomEmbedding(384))

    // Second engine on same path should succeed (operation-level locking)
    const engine2 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(engine2)

    // Both engines should see the same data
    expect(engine1.hasKey('first')).toBe(true)
    expect(engine2.hasKey('first')).toBe(true)
  })

  it('two sequential writes from same process both succeed', async () => {
    const paths = createTestPaths('sequential-writes')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(engine)

    const embedding1 = generateRandomEmbedding(384)
    const embedding2 = generateRandomEmbedding(384)

    await engine.writeRecord('key1', embedding1)
    await engine.writeRecord('key2', embedding2)

    expect(engine.count()).toBe(2)
    expect(engine.hasKey('key1')).toBe(true)
    expect(engine.hasKey('key2')).toBe(true)

    const record1 = await engine.readRecord('key1')
    const record2 = await engine.readRecord('key2')
    expect(embeddingsEqual(record1!.embedding, embedding1)).toBe(true)
    expect(embeddingsEqual(record2!.embedding, embedding2)).toBe(true)
  })

  it('close is idempotent', async () => {
    const paths = createTestPaths('locking-close-twice')
    testPathsList.push(paths)

    const engine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    await engine.writeRecord('test', generateRandomEmbedding(384))
    await engine.close()
    await engine.close() // Should not throw

    // Can reopen after double close
    const engine2 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(engine2)

    expect(engine2.hasKey('test')).toBe(true)
  })

  it('data persists after close and reopen', async () => {
    const paths = createTestPaths('locking-persist')
    testPathsList.push(paths)

    const embedding = generateRandomEmbedding(384)

    const engine1 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    await engine1.writeRecord('persistent', embedding)
    await engine1.close()

    const engine2 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(engine2)

    const record = await engine2.readRecord('persistent')
    expect(record).not.toBeNull()
    expect(embeddingsEqual(record!.embedding, embedding)).toBe(true)
  })

  it('multiple sequential open/close cycles work correctly', async () => {
    const paths = createTestPaths('locking-cycles')
    testPathsList.push(paths)

    const embeddings: Float32Array[] = []

    // Multiple cycles
    for (let i = 0; i < 5; i++) {
      const engine = await StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384
      })

      const embedding = generateRandomEmbedding(384)
      embeddings.push(embedding)
      await engine.writeRecord(`key${i}`, embedding)
      await engine.close()
    }

    // Final verification
    const finalEngine = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(finalEngine)

    expect(finalEngine.count()).toBe(5)
    for (let i = 0; i < 5; i++) {
      const record = await finalEngine.readRecord(`key${i}`)
      expect(record).not.toBeNull()
      expect(embeddingsEqual(record!.embedding, embeddings[i])).toBe(true)
    }
  })

  it('lock is released after write even when dimension error occurs', async () => {
    const paths = createTestPaths('locking-error-release')
    testPathsList.push(paths)

    const engine1 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(engine1)

    await engine1.writeRecord('test', generateRandomEmbedding(384))

    // Simulate an error by passing wrong dimension - should be rejected
    try {
      await engine1.writeRecord('bad', generateRandomEmbedding(768))
    } catch {
      // Expected to fail
    }

    // Lock should be released (no session lock to release)
    // Can still write with another engine instance
    const engine2 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(engine2)

    expect(engine2.hasKey('test')).toBe(true)
    await engine2.writeRecord('another', generateRandomEmbedding(384))
    expect(engine2.hasKey('another')).toBe(true)
  })
})
