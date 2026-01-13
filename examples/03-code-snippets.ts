/* eslint-disable no-console */
/**
 * Example: Code Snippet Library
 *
 * This example demonstrates searching code snippets by natural language
 * descriptions. Find code by what it does rather than exact keywords.
 *
 * Use case: Developer documentation, code example libraries,
 * or internal snippet repositories.
 *
 * Run: bun run examples/03-code-snippets.ts
 */

import { unlink } from 'node:fs/promises'

import { EmbeddingEngine } from '../src'

// Sample code snippets with descriptions
const snippets = [
  {
    id: 'read-file-chunks',
    description:
      'Read a file in chunks asynchronously to handle large files without loading everything into memory. Uses streams and async iteration.',
    language: 'TypeScript',
    code: `async function* readFileInChunks(path: string, chunkSize = 64 * 1024) {
  const file = Bun.file(path)
  const stream = file.stream()
  const reader = stream.getReader()

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      yield value
    }
  } finally {
    reader.releaseLock()
  }
}`
  },
  {
    id: 'debounce-function',
    description:
      'Create a debounced function that delays execution until after a specified wait period has elapsed since the last call. Useful for rate-limiting API calls or search inputs.',
    language: 'JavaScript',
    code: `function debounce(func, wait) {
  let timeout
  return function(...args) {
    clearTimeout(timeout)
    timeout = setTimeout(() => func.apply(this, args), wait)
  }
}`
  },
  {
    id: 'retry-with-backoff',
    description:
      'Retry an async function with exponential backoff. Automatically retries failed operations with increasing delays between attempts.',
    language: 'TypeScript',
    code: `async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  baseDelay = 1000
): Promise<T> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn()
    } catch (error) {
      if (i === maxRetries - 1) throw error
      const delay = baseDelay * Math.pow(2, i)
      await new Promise((resolve) => setTimeout(resolve, delay))
    }
  }
  throw new Error('Max retries exceeded')
}`
  },
  {
    id: 'deep-clone',
    description:
      'Deep clone an object including nested objects and arrays. Handles circular references and preserves prototype chains.',
    language: 'JavaScript',
    code: `function deepClone(obj, hash = new WeakMap()) {
  if (Object(obj) !== obj) return obj
  if (hash.has(obj)) return hash.get(obj)

  const result = Array.isArray(obj)
    ? []
    : obj.constructor
    ? new obj.constructor()
    : Object.create(null)

  hash.set(obj, result)

  return Object.assign(
    result,
    ...Object.keys(obj).map((key) => ({
      [key]: deepClone(obj[key], hash)
    }))
  )
}`
  },
  {
    id: 'throttle-function',
    description:
      'Create a throttled function that only executes once per specified time period. Different from debounce - guarantees execution at regular intervals.',
    language: 'JavaScript',
    code: `function throttle(func, limit) {
  let inThrottle
  return function(...args) {
    if (!inThrottle) {
      func.apply(this, args)
      inThrottle = true
      setTimeout(() => (inThrottle = false), limit)
    }
  }
}`
  },
  {
    id: 'batch-promises',
    description:
      'Process an array of async tasks in batches with a concurrency limit. Prevents overwhelming APIs or system resources.',
    language: 'TypeScript',
    code: `async function batchProcess<T, R>(
  items: T[],
  processor: (item: T) => Promise<R>,
  concurrency = 5
): Promise<R[]> {
  const results: R[] = []
  for (let i = 0; i < items.length; i += concurrency) {
    const batch = items.slice(i, i + concurrency)
    const batchResults = await Promise.all(batch.map(processor))
    results.push(...batchResults)
  }
  return results
}`
  }
]

async function main() {
  const dbPath = './examples-code-snippets.jsonl'
  const engine = new EmbeddingEngine({ storePath: dbPath })

  console.log('💻 Code Snippet Library Example\n')
  console.log('📚 Indexing code snippets...')

  // Store snippet descriptions in batch (not the code itself)
  await engine.storeMany(
    snippets.map((snippet) => ({ key: snippet.id, text: snippet.description }))
  )

  console.log(`✓ Indexed ${snippets.length} code snippets\n`)

  // Example searches using natural language
  const searches = [
    'How do I limit how often a function runs?',
    'I need to handle large files without running out of memory',
    'Retry failed network requests automatically',
    'Copy an object with all its nested properties'
  ]

  for (const query of searches) {
    console.log(`\n🔍 Search: "${query}"`)
    console.log('─'.repeat(60))

    const results = await engine.search(query, 2, 0.4)

    if (results.length === 0) {
      console.log('No matching snippets found.')
      continue
    }

    // Show the best match with full code
    const topResult = results[0]
    const snippet = snippets.find((s) => s.id === topResult.key)

    if (snippet) {
      console.log(`\n✨ Best match: ${snippet.id}`)
      console.log(`   Similarity: ${topResult.similarity.toFixed(3)}`)
      console.log(`   Language: ${snippet.language}`)
      console.log('\n' + snippet.description)
      console.log('\n```' + snippet.language.toLowerCase())
      console.log(snippet.code)
      console.log('```')

      // Show alternative if available
      if (results.length > 1) {
        const altSnippet = snippets.find((s) => s.id === results[1].key)
        if (altSnippet) {
          console.log(
            `\nAlternative: ${altSnippet.id} (${results[1].similarity.toFixed(3)})`
          )
          console.log(`   ${altSnippet.description}`)
        }
      }
    }
  }

  console.log('\n\n💡 Benefits:')
  console.log('─'.repeat(60))
  console.log('• Search by what code does, not exact keywords')
  console.log('• Find solutions to problems described in plain English')
  console.log('• Discover related code patterns you might not have considered')
  console.log('• Build an intelligent code documentation system')

  // Cleanup
  console.log('\n🧹 Cleaning up...')
  await engine.dispose()
  await unlink(dbPath).catch(() => {})
  console.log('✓ Done!')
}

main().catch(console.error)
