/* eslint-disable no-console */
/**
 * Example: Content Recommendation
 *
 * This example demonstrates "more like this" functionality by finding
 * semantically similar content items.
 *
 * Use case: Blog post recommendations, product suggestions,
 * related content discovery, or personalized feeds.
 *
 * Run: bun run examples/04-content-recommendation.ts
 */

import { unlink } from 'node:fs/promises'

import { EmbeddingEngine } from '../src'

// Sample blog posts
const blogPosts = [
  {
    id: 'post-1',
    title: 'Getting Started with Bun: A Modern JavaScript Runtime',
    description:
      'Learn how Bun provides faster package management, built-in TypeScript support, and improved developer experience compared to Node.js. Includes benchmarks and migration tips.',
    category: 'JavaScript',
    tags: ['bun', 'javascript', 'performance']
  },
  {
    id: 'post-2',
    title: 'Building Semantic Search with Vector Embeddings',
    description:
      'A practical guide to implementing semantic search using text embeddings and cosine similarity. Covers vector databases, embedding models, and real-world applications.',
    category: 'Machine Learning',
    tags: ['embeddings', 'search', 'ml']
  },
  {
    id: 'post-3',
    title: 'TypeScript 5.0: New Features and Breaking Changes',
    description:
      'Explore the latest TypeScript release including decorators, const type parameters, and performance improvements. Migration guide included for existing projects.',
    category: 'JavaScript',
    tags: ['typescript', 'javascript', 'types']
  },
  {
    id: 'post-4',
    title: 'Introduction to RAG: Retrieval Augmented Generation',
    description:
      'Understand how RAG combines large language models with external knowledge bases to provide accurate, up-to-date answers. Includes implementation patterns and best practices.',
    category: 'Machine Learning',
    tags: ['llm', 'rag', 'ai']
  },
  {
    id: 'post-5',
    title: 'Optimizing Node.js Performance with Clustering',
    description:
      'Learn how to leverage multiple CPU cores in Node.js using the cluster module. Includes load balancing strategies and monitoring techniques.',
    category: 'JavaScript',
    tags: ['nodejs', 'performance', 'scaling']
  },
  {
    id: 'post-6',
    title: 'Understanding Transformer Models in NLP',
    description:
      'Deep dive into transformer architecture, attention mechanisms, and how they revolutionized natural language processing. Covers BERT, GPT, and their applications.',
    category: 'Machine Learning',
    tags: ['nlp', 'transformers', 'deep-learning']
  },
  {
    id: 'post-7',
    title: 'API Design Best Practices with REST and GraphQL',
    description:
      'Compare REST and GraphQL for API design. Learn when to use each approach, common pitfalls, and how to build scalable, maintainable APIs.',
    category: 'Backend',
    tags: ['api', 'rest', 'graphql']
  },
  {
    id: 'post-8',
    title: 'Building Real-time Applications with WebSockets',
    description:
      'Create interactive real-time features using WebSockets. Covers connection management, scaling considerations, and fallback strategies for older browsers.',
    category: 'Backend',
    tags: ['websockets', 'realtime', 'networking']
  }
]

async function main() {
  const dbPath = './examples-recommendations.jsonl'
  const engine = new EmbeddingEngine({ storePath: dbPath })

  console.log('🎯 Content Recommendation Example\n')
  console.log('📝 Indexing blog posts...')

  // Store combined title + description in batch for better semantic matching
  await engine.storeMany(
    blogPosts.map((post) => ({
      key: post.id,
      text: `${post.title}. ${post.description}`
    }))
  )

  console.log(`✓ Indexed ${blogPosts.length} blog posts\n`)

  // Pick a few posts to show recommendations for
  const examplePosts = ['post-2', 'post-5', 'post-7']

  for (const postId of examplePosts) {
    const currentPost = blogPosts.find((p) => p.id === postId)
    if (!currentPost) {
      continue
    }

    console.log('\n' + '═'.repeat(60))
    console.log(`📖 Current Post: "${currentPost.title}"`)
    console.log(`   Category: ${currentPost.category}`)
    console.log('═'.repeat(60))

    // Create search query from current post
    const searchText = `${currentPost.title}. ${currentPost.description}`

    // Search for similar posts (get 4 so we can exclude the current one)
    const results = await engine.search(searchText, 4, 0.3)

    console.log('\n🔗 Recommended Posts:\n')

    let recommendationCount = 0
    for (const result of results) {
      // Skip the current post itself
      if (result.key === postId) {
        continue
      }

      const recommendedPost = blogPosts.find((p) => p.id === result.key)

      if (!recommendedPost) {
        continue
      }

      recommendationCount++
      console.log(`${recommendationCount}. ${recommendedPost.title}`)
      console.log(
        `   Similarity: ${result.similarity.toFixed(3)} | Category: ${recommendedPost.category}`
      )
      console.log(`   ${recommendedPost.description.substring(0, 80)}...`)
      console.log()

      // Stop after 3 recommendations
      if (recommendationCount >= 3) break
    }
  }

  console.log('\n' + '═'.repeat(60))
  console.log('💡 Recommendation Strategies')
  console.log('═'.repeat(60))
  console.log('1. Content-based: Find similar posts by semantic similarity')
  console.log('2. Diversity: Mix categories while maintaining relevance')
  console.log('3. Threshold tuning: Adjust minSimilarity to control relevance')
  console.log('4. Hybrid approach: Combine with user behavior data')
  console.log()
  console.log('💡 Real-world Applications')
  console.log('─'.repeat(60))
  console.log('• "More like this" sections on blogs')
  console.log('• Product recommendations in e-commerce')
  console.log('• Related articles in news sites')
  console.log('• Similar content in video platforms')
  console.log('• Discover weekly playlists in music apps')

  // Cleanup
  console.log('\n🧹 Cleaning up...')
  await engine.dispose()
  await unlink(dbPath).catch(() => {})
  console.log('✓ Done!')
}

main().catch(console.error)
