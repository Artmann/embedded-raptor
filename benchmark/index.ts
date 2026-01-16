import { existsSync, unlinkSync } from 'fs'
import { join } from 'path'

import { EmbeddingEngine } from '../src/engine'
import { getAllQueries, getQuickQueries } from './queries'
import { runGetBenchmarks } from './suites/get-benchmark'
import { runSearchBenchmarks } from './suites/search-benchmark'
import { runStoreBenchmarks } from './suites/store-benchmark'
import { createEmbeddingText, loadArticles } from './utils/data-loader'
import type { Article } from './utils/data-loader'
import {
  formatDuration,
  printModernSection,
  printMarkdownTable
} from './utils/reporter'

interface BenchmarkOptions {
  quick: boolean
  storeSizes: number[]
  readSizes: number[]
  dataDir: string
}

/**
 * Parse command line arguments
 */
function parseArgs(): BenchmarkOptions {
  const args = process.argv.slice(2)

  const quick = args.includes('--quick')
  const dataDirArg = args.find((arg) => arg.startsWith('--data-dir='))
  const dataDir = dataDirArg
    ? dataDirArg.split('=')[1]
    : join(__dirname, 'data')

  // Store sizes - kept smaller because storeMany is still slow
  const defaultStoreSizes = [100, 500, 1000, 2000]
  const quickStoreSizes = [100]

  // Read sizes - larger because search/get are faster
  // For 5k/10k, download more sample data to benchmark/data
  const defaultReadSizes = [100, 500, 1000, 2000, 5000, 10000]
  const quickReadSizes = [100]

  return {
    quick,
    storeSizes: quick ? quickStoreSizes : defaultStoreSizes,
    readSizes: quick ? quickReadSizes : defaultReadSizes,
    dataDir
  }
}

/**
 * Clean up benchmark database files
 */
function cleanupFiles(baseStorePath: string, sizes: number[]): void {
  for (const size of sizes) {
    // Clean up all file formats: .raptor, .raptor-wal, .raptor.lock, and legacy .jsonl
    const extensions = ['.raptor', '.raptor-wal', '.raptor.lock', '.jsonl']
    for (const ext of extensions) {
      const filePath = `${baseStorePath}-${size}${ext}`
      if (existsSync(filePath)) {
        unlinkSync(filePath)
      }
    }
  }
}

/**
 * Create database for a specific size (used for larger read-only datasets)
 */
async function createDatabase(
  articles: Article[],
  size: number,
  baseStorePath: string
): Promise<void> {
  const storePath = `${baseStorePath}-${size}.jsonl`

  // Skip if already exists (from store benchmarks)
  if (existsSync(storePath)) {
    return
  }

  console.log(`  Creating database with ${size} documents...`)
  const engine = new EmbeddingEngine({ storePath })

  const items = articles.slice(0, size).map((article) => ({
    key: article.id,
    text: createEmbeddingText(article)
  }))

  const startTime = performance.now()
  await engine.storeMany(items)
  const duration = performance.now() - startTime

  await engine.dispose()

  console.log(`    Created in ${formatDuration(duration)}`)
}

/**
 * Main benchmark runner
 */
async function main(): Promise<void> {
  const options = parseArgs()

  printModernSection('Raptor Benchmark Suite')
  console.log(`Mode: ${options.quick ? 'Quick' : 'Comprehensive'}`)
  console.log(`Store sizes: ${options.storeSizes.join(', ')}`)
  console.log(`Read sizes: ${options.readSizes.join(', ')}`)
  console.log(`Data directory: ${options.dataDir}`)
  console.log()

  // Load articles - need enough for largest read size
  const maxSize = Math.max(...options.readSizes)
  console.log('Loading articles...')

  // Load articles from all subdirectories in the data folder
  const { readdirSync } = await import('fs')
  const dataDirs = readdirSync(options.dataDir, { withFileTypes: true })
    .filter((dirent) => dirent.isDirectory())
    .map((dirent) => join(options.dataDir, dirent.name))

  let allArticles: Article[] = []
  for (const dir of dataDirs) {
    const articles = loadArticles(dir, maxSize - allArticles.length)
    allArticles = allArticles.concat(articles)
    if (allArticles.length >= maxSize) break
  }

  console.log(
    `Loaded ${allArticles.length} articles from ${dataDirs.length} directories`
  )

  // Filter sizes to only include those we have data for
  const availableCount = allArticles.length
  const actualStoreSizes = options.storeSizes.filter(
    (size) => size <= availableCount
  )
  const actualReadSizes = options.readSizes.filter(
    (size) => size <= availableCount
  )

  if (allArticles.length < maxSize) {
    console.log(
      `Warning: Only ${allArticles.length} articles available, but ${maxSize} requested.`
    )
    console.log(
      `Benchmarks will run with: Store sizes: ${actualStoreSizes.join(', ')}, Read sizes: ${actualReadSizes.join(', ')}`
    )
    console.log(
      `To test larger datasets, download more articles to benchmark/data subdirectories.`
    )
  }
  console.log()

  // Get queries
  const queries = options.quick ? getQuickQueries() : getAllQueries()
  console.log(`Using ${queries.length} test queries`)
  console.log()

  const baseStorePath = join(__dirname, '.tmp', 'benchmark')

  try {
    // 1. Store Benchmarks
    printModernSection('Store Operations (Bulk Insert)')
    const storeResults = await runStoreBenchmarks(
      allArticles,
      actualStoreSizes,
      baseStorePath
    )

    // Print store results table
    printMarkdownTable(
      [
        { header: 'Size', align: 'right' },
        { header: 'Duration', align: 'right' },
        { header: 'Throughput', align: 'right' },
        { header: 'Avg/Doc', align: 'right' }
      ],
      storeResults.map((r) => ({
        Size: r.datasetSize,
        Duration: formatDuration(r.totalDurationMs),
        Throughput: r.throughput,
        'Avg/Doc': formatDuration(r.avgDurationPerDocMs)
      }))
    )

    // 1b. Create databases for larger read sizes
    const additionalSizes = actualReadSizes.filter(
      (size) => !actualStoreSizes.includes(size)
    )
    if (additionalSizes.length > 0) {
      printModernSection('Creating Additional Databases for Read Benchmarks')
      for (const size of additionalSizes) {
        await createDatabase(allArticles, size, baseStorePath)
      }
    }

    // 2. Search Benchmarks
    printModernSection('Search Operations')
    const searchResults = await runSearchBenchmarks(
      baseStorePath,
      actualReadSizes,
      queries
    )

    // Print search results table
    printMarkdownTable(
      [
        { header: 'Size', align: 'right' },
        { header: 'Cold Start', align: 'right' },
        { header: 'Avg Warm', align: 'right' }
      ],
      searchResults.map((r) => ({
        Size: r.datasetSize,
        'Cold Start': formatDuration(r.coldStartMs),
        'Avg Warm': formatDuration(r.avgWarmMs)
      }))
    )

    // 3. Get Benchmarks
    printModernSection('Get Operations')
    const getResults = await runGetBenchmarks(
      baseStorePath,
      allArticles,
      actualReadSizes
    )

    // Print get results table
    printMarkdownTable(
      [
        { header: 'Size', align: 'right' },
        { header: 'Sequential', align: 'right' },
        { header: 'Random', align: 'right' }
      ],
      getResults.map((r) => ({
        Size: r.datasetSize,
        Sequential: formatDuration(r.sequentialAvgMs),
        Random: formatDuration(r.randomAvgMs)
      }))
    )

    // 4. Scalability Analysis
    printModernSection('Scalability Analysis')

    // Calculate scalability metrics
    if (storeResults.length >= 2) {
      console.log('Store Operation Scaling:')
      for (let i = 1; i < storeResults.length; i++) {
        const prev = storeResults[i - 1]
        const curr = storeResults[i]
        const sizeRatio = curr.datasetSize / prev.datasetSize
        const timeRatio = curr.totalDurationMs / prev.totalDurationMs
        const efficiency = (timeRatio / sizeRatio) * 100

        console.log(
          `  ${prev.datasetSize} → ${curr.datasetSize}: ` +
            `${sizeRatio.toFixed(1)}x size, ${timeRatio.toFixed(1)}x time ` +
            `(${efficiency.toFixed(0)}% efficiency)`
        )
      }
      console.log()
    }

    if (searchResults.length >= 2) {
      console.log('Search Operation Scaling:')
      for (let i = 1; i < searchResults.length; i++) {
        const prev = searchResults[i - 1]
        const curr = searchResults[i]
        const sizeRatio = curr.datasetSize / prev.datasetSize
        const timeRatio = curr.avgWarmMs / prev.avgWarmMs
        const efficiency = (timeRatio / sizeRatio) * 100

        console.log(
          `  ${prev.datasetSize} → ${curr.datasetSize}: ` +
            `${sizeRatio.toFixed(1)}x size, ${timeRatio.toFixed(1)}x time ` +
            `(${efficiency.toFixed(0)}% efficiency)`
        )
      }
      console.log()
    }

    // 5. Summary
    printModernSection('Summary')
    console.log('Key Insights:')
    console.log()

    const largestStore = storeResults[storeResults.length - 1]
    console.log(
      `• Store throughput: ${largestStore.throughput} (${formatDuration(largestStore.avgDurationPerDocMs)}/doc)`
    )

    const largestSearch = searchResults[searchResults.length - 1]
    console.log(
      `• Search latency (${largestSearch.datasetSize} docs): ${formatDuration(largestSearch.avgWarmMs)} avg`
    )

    const largestGet = getResults[getResults.length - 1]
    console.log(
      `• Get latency (${largestGet.datasetSize} docs): ${formatDuration(largestGet.sequentialAvgMs)} sequential, ` +
        `${formatDuration(largestGet.randomAvgMs)} random`
    )

    console.log()
    console.log('Benchmark completed successfully!')
  } finally {
    // Cleanup - clean up all database files (both store and read sizes)
    const allSizes = Array.from(
      new Set([...actualStoreSizes, ...actualReadSizes])
    )
    console.log()
    console.log('Cleaning up benchmark files...')
    cleanupFiles(baseStorePath, allSizes)
  }
}

// Run the benchmark
main().catch((error) => {
  console.error('Benchmark failed:', error)
  process.exit(1)
})
