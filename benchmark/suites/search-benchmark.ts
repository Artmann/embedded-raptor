import { EmbeddingEngine } from '../../src/engine'
import type { TestQuery } from '../queries'
import { formatDuration } from '../utils/reporter'

export interface SearchBenchmarkResult {
  datasetSize: number
  numQueries: number
  coldStartMs: number
  avgWarmMs: number
}

/**
 * Benchmark search operations
 */
export async function benchmarkSearch(
  storePath: string,
  queries: TestQuery[],
  datasetSize: number
): Promise<SearchBenchmarkResult> {
  const engine = new EmbeddingEngine({ storePath })

  // Cold start - first query
  const coldStartTime = performance.now()
  await engine.search(queries[0].query, 5, 0.5)
  const coldStartMs = performance.now() - coldStartTime

  // Warm queries - remaining queries
  const warmStartTime = performance.now()

  for (let i = 1; i < queries.length; i++) {
    await engine.search(queries[i].query, 5, 0.5)
  }

  const totalWarmDuration = performance.now() - warmStartTime
  const avgWarmMs =
    queries.length > 1 ? totalWarmDuration / (queries.length - 1) : 0

  await engine.dispose()

  return {
    datasetSize,
    numQueries: queries.length,
    coldStartMs,
    avgWarmMs
  }
}

/**
 * Run search benchmarks for multiple dataset sizes
 */
export async function runSearchBenchmarks(
  baseStorePath: string,
  sizes: number[],
  queries: TestQuery[]
): Promise<SearchBenchmarkResult[]> {
  const results: SearchBenchmarkResult[] = []

  for (let i = 0; i < sizes.length; i++) {
    const size = sizes[i]
    const storePath = `${baseStorePath}-${size}.jsonl`

    console.log(
      `  Benchmarking search with ${size} documents (${queries.length} queries)... (${i + 1}/${sizes.length})`
    )

    const result = await benchmarkSearch(storePath, queries, size)
    results.push(result)

    console.log(
      `    Cold: ${formatDuration(result.coldStartMs)}, ` +
        `Warm Avg: ${formatDuration(result.avgWarmMs)}`
    )
  }

  return results
}
