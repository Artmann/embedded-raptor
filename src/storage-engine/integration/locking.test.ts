/* eslint-disable @typescript-eslint/no-non-null-assertion */
/**
 * Database locking integration tests for StorageEngine.
 * Tests exclusive file locking behavior.
 */

import { describe, it, expect, afterEach } from 'vitest'
import { StorageEngine } from '../storage-engine'
import { DatabaseLockedError } from '../file-lock'
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

  it('storage engine opens successfully with lock', async () => {
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

  it('second engine instance on same path throws DatabaseLockedError', async () => {
    const paths = createTestPaths('locking-conflict')
    testPathsList.push(paths)

    const engine1 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(engine1)

    await engine1.writeRecord('first', generateRandomEmbedding(384))

    // Second engine should fail to acquire lock immediately (lockTimeout: 0)
    await expect(
      StorageEngine.create({
        dataPath: paths.dataPath,
        dimension: 384,
        lockTimeout: 0
      })
    ).rejects.toThrow(DatabaseLockedError)
  })

  it('after close, another instance can open', async () => {
    const paths = createTestPaths('locking-reopen')
    testPathsList.push(paths)

    const embedding1 = generateRandomEmbedding(384)
    const embedding2 = generateRandomEmbedding(384)

    const engine1 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    await engine1.writeRecord('first', embedding1)
    await engine1.close()

    const engine2 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(engine2)

    // Engine2 should be able to acquire lock now
    await engine2.writeRecord('second', embedding2)

    expect(engine2.count()).toBe(2)
    expect(engine2.hasKey('first')).toBe(true)
    expect(engine2.hasKey('second')).toBe(true)
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

  it('lock is released on engine close even after errors', async () => {
    const paths = createTestPaths('locking-error-release')
    testPathsList.push(paths)

    const engine1 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })

    await engine1.writeRecord('test', generateRandomEmbedding(384))

    // Simulate an error by passing wrong dimension - should be rejected
    try {
      await engine1.writeRecord('bad', generateRandomEmbedding(768))
    } catch {
      // Expected to fail
    }

    await engine1.close()

    // Lock should be released, can open new engine
    const engine2 = await StorageEngine.create({
      dataPath: paths.dataPath,
      dimension: 384
    })
    engines.push(engine2)

    expect(engine2.hasKey('test')).toBe(true)
  })
})
