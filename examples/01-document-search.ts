/* eslint-disable no-console */
/**
 * Example: Document Search / Simple RAG
 *
 * This example demonstrates semantic search over documentation chunks,
 * similar to a basic RAG (Retrieval Augmented Generation) pattern.
 *
 * Use case: Search a knowledge base with natural language queries
 * and retrieve the most relevant documentation sections.
 *
 * Run: bun run examples/01-document-search.ts
 */

import { unlink } from 'node:fs/promises'

import { EmbeddingEngine } from '../src'

// Sample documentation chunks about JavaScript
const docs = [
  {
    key: 'js-promises',
    text: 'Promises in JavaScript represent the eventual completion or failure of an asynchronous operation. They provide a cleaner alternative to callbacks and support chaining with .then() and .catch() methods.'
  },
  {
    key: 'js-async-await',
    text: 'Async/await is syntactic sugar built on top of promises. The async keyword makes a function return a promise, while await pauses execution until a promise resolves, making asynchronous code look synchronous.'
  },
  {
    key: 'js-closures',
    text: 'A closure is a function that has access to variables in its outer lexical scope, even after the outer function has returned. Closures are commonly used for data privacy and creating function factories.'
  },
  {
    key: 'js-event-loop',
    text: 'The event loop is the mechanism that handles asynchronous callbacks in JavaScript. It continuously checks the call stack and callback queue, executing callbacks when the stack is empty.'
  },
  {
    key: 'js-prototypes',
    text: 'JavaScript uses prototypal inheritance where objects can inherit properties and methods from other objects. Every object has an internal prototype link, accessible via __proto__ or Object.getPrototypeOf().'
  },
  {
    key: 'js-modules',
    text: 'ES6 modules allow you to split code into reusable files using export and import statements. Modules have their own scope and are loaded asynchronously, improving code organization and performance.'
  }
]

async function main() {
  const dbPath = './examples-doc-search.jsonl'
  const engine = new EmbeddingEngine({ storePath: dbPath })

  console.log('🔍 Document Search Example\n')
  console.log('📚 Storing documentation chunks...')

  // Store all documentation chunks in batch
  await engine.storeMany(docs)

  console.log(`✓ Stored ${docs.length} documentation chunks\n`)

  // Example queries
  const queries = [
    'How do I handle asynchronous code in JavaScript?',
    'What is inheritance in JavaScript?',
    'How does JavaScript handle concurrency?'
  ]

  for (const query of queries) {
    console.log(`\n❓ Query: "${query}"`)
    console.log('─'.repeat(60))

    // Search with minimum similarity of 0.3, return top 3 results
    const results = await engine.search(query, 3, 0.3)

    if (results.length === 0) {
      console.log('No relevant documents found.')
      continue
    }

    // Display results with context
    for (let i = 0; i < results.length; i++) {
      const result = results[i]
      const doc = docs.find((d) => d.key === result.key)

      console.log(
        `\n${i + 1}. ${result.key} (similarity: ${result.similarity.toFixed(3)})`
      )

      if (doc) {
        // Show first 150 characters of the document
        const preview =
          doc.text.length > 150 ? doc.text.substring(0, 150) + '...' : doc.text
        console.log(`   ${preview}`)
      }
    }
  }

  console.log('\n\n💡 Use case: Retrieve context for RAG')
  console.log('─'.repeat(60))
  console.log('In a RAG system, you would:')
  console.log('1. Search for relevant documents (like above)')
  console.log('2. Retrieve the full text of top results')
  console.log('3. Pass them as context to an LLM prompt')
  console.log('4. Get an answer grounded in your knowledge base')

  // Cleanup
  console.log('\n🧹 Cleaning up...')
  await engine.dispose()
  await unlink(dbPath).catch(() => {})
  console.log('✓ Done!')
}

main().catch(console.error)
