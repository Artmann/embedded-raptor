import { EmbeddingEngine } from '../../src/engine'
import type { Article } from '../utils/data-loader'
import { formatDuration } from '../utils/reporter'

export interface GetBenchmarkResult {
  datasetSize: number
  sequentialAvgMs: number
  randomAvgMs: number
}

/**
 * Benchmark get operations with sequential access pattern
 */
async function benchmarkSequentialGet(
  engine: EmbeddingEngine,
  articleIds: string[],
  sampleSize: number
): Promise<{ avgMs: number }> {
  // Take first N articles for sequential access
  const idsToTest = articleIds.slice(0, sampleSize)

  const startTime = performance.now()

  for (const id of idsToTest) {
    await engine.get(id)
  }

  const totalDuration = performance.now() - startTime

  return {
    avgMs: totalDuration / idsToTest.length
  }
}

/**
 * Benchmark get operations with random access pattern
 */
async function benchmarkRandomGet(
  engine: EmbeddingEngine,
  articleIds: string[],
  sampleSize: number
): Promise<{ avgMs: number }> {
  const shuffled = [...articleIds].sort(() => (Math.random() - 0.5 ? 1 : -1))
  const idsToTest = shuffled.slice(0, sampleSize)

  const startTime = performance.now()

  for (const id of idsToTest) {
    await engine.get(id)
  }

  const totalDuration = performance.now() - startTime

  return {
    avgMs: totalDuration / idsToTest.length
  }
}

/**
 * Benchmark get operations
 */
export async function benchmarkGet(
  storePath: string,
  articles: Article[],
  sampleSize: number = 20
): Promise<GetBenchmarkResult> {
  const engine = new EmbeddingEngine({ storePath })
  const articleIds = articles.map((a) => a.id)

  const sequential = await benchmarkSequentialGet(
    engine,
    articleIds,
    sampleSize
  )

  const random = await benchmarkRandomGet(engine, articleIds, sampleSize)

  await engine.dispose()

  return {
    datasetSize: articles.length,
    sequentialAvgMs: sequential.avgMs,
    randomAvgMs: random.avgMs
  }
}

/**
 * Run get benchmarks for multiple dataset sizes
 */
export async function runGetBenchmarks(
  baseStorePath: string,
  allArticles: Article[],
  sizes: number[]
): Promise<GetBenchmarkResult[]> {
  const results: GetBenchmarkResult[] = []

  for (let i = 0; i < sizes.length; i++) {
    const size = sizes[i]
    const storePath = `${baseStorePath}-${size}.jsonl`
    const articles = allArticles.slice(0, size)

    console.log(
      `  Benchmarking get with ${size} documents... (${i + 1}/${sizes.length})`
    )

    const result = await benchmarkGet(storePath, articles)

    results.push(result)

    console.log(
      `    Sequential: ${formatDuration(result.sequentialAvgMs)}, ` +
        `Random: ${formatDuration(result.randomAvgMs)}`
    )
  }

  return results
}
