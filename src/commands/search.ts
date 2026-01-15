import { command } from 'cleye'
import { EmbeddingEngine } from '../engine'
import { sharedFlags, searchFlags } from './flags'

export const search = command(
  {
    name: 'search',
    description: 'Search for similar embeddings using a query',
    parameters: ['<query>'],
    flags: {
      ...sharedFlags,
      ...searchFlags
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
