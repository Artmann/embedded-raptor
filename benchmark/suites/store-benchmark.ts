import { EmbeddingEngine } from '../../src/engine'
import type { Article } from '../utils/data-loader'
import { createEmbeddingText } from '../utils/data-loader'
import { formatDuration, formatThroughput } from '../utils/reporter'

export interface StoreBenchmarkResult {
  avgDurationPerDocMs: number
  datasetSize: number
  throughput: string
  totalDurationMs: number
}

/**
 * Benchmark store operations (bulk insert)
 */
export async function benchmarkStore(
  articles: Article[],
  storePath: string
): Promise<StoreBenchmarkResult> {
  const engine = new EmbeddingEngine({ storePath })

  const items = articles.map((article) => ({
    key: article.id,
    text: createEmbeddingText(article)
  }))

  const startTime = performance.now()

  await engine.storeMany(items)

  const endTime = performance.now()

  await engine.dispose()

  const totalDurationMs = endTime - startTime
  const avgDurationPerDocMs =
    articles.length > 0 ? totalDurationMs / articles.length : 0

  return {
    avgDurationPerDocMs,
    datasetSize: articles.length,
    throughput: formatThroughput(articles.length, totalDurationMs),
    totalDurationMs
  }
}

/**
 * Run store benchmarks for multiple dataset sizes
 */
export async function runStoreBenchmarks(
  allArticles: Article[],
  sizes: number[],
  baseStorePath: string
): Promise<StoreBenchmarkResult[]> {
  const results: StoreBenchmarkResult[] = []

  for (let i = 0; i < sizes.length; i++) {
    const size = sizes[i]
    const articles = allArticles.slice(0, size)
    const storePath = `${baseStorePath}-${size}.jsonl`

    console.log(
      `  Benchmarking store with ${size} documents... (${i + 1}/${sizes.length})`
    )

    const result = await benchmarkStore(articles, storePath)

    results.push(result)

    console.log(
      `    Duration: ${formatDuration(result.totalDurationMs)}, ` +
        `Throughput: ${result.throughput}`
    )
  }

  return results
}
