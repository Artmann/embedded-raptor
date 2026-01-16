import { command } from 'cleye'
import { EmbeddingEngine } from '../engine'
import { sharedFlags, searchFlags } from './flags'

export const search = command(
  {
    name: 'search',
    parameters: ['<query>'],
    flags: {
      ...sharedFlags,
      ...searchFlags
    },
    help: {
      description: 'Search for similar embeddings using a query',
      examples: [
        'raptor search "artificial intelligence"',
        'raptor search -l 5 -m 0.7 "machine learning"',
        'raptor search -s ./my-db.raptor "find similar docs"'
      ]
    }
  },
  async (argv) => {
    const engine = new EmbeddingEngine({
      storePath: argv.flags.storePath
    })

    try {
      const [query] = argv._
      const results = await engine.search(
        query,
        argv.flags.limit,
        argv.flags.minSimilarity
      )

      if (results.length === 0) {
        console.log('No results found')
      } else {
        console.log(`Found ${results.length} result(s):\n`)
        for (const result of results) {
          console.log(`${result.key}: ${result.similarity.toFixed(6)}`)
        }
      }
    } finally {
      await engine.dispose()
    }
  }
)
