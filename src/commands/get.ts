import { command } from 'cleye'
import { EmbeddingEngine } from '../engine'
import { sharedFlags } from './flags'

export const get = command(
  {
    name: 'get',
    parameters: ['<key>'],
    flags: {
      ...sharedFlags
    },
    help: {
      description: 'Retrieve an embedding entry by key',
      examples: [
        'raptor get doc1',
        'raptor get -s ./my-db.raptor mykey'
      ]
    }
  },
  async (argv) => {
    const engine = new EmbeddingEngine({
      storePath: argv.flags.storePath
    })

    try {
      const [key] = argv._
      const entry = await engine.get(key)

      if (entry) {
        console.log(
          JSON.stringify(
            {
              key: entry.key,
              embeddingDimensions: entry.embedding.length,
              timestamp: new Date(entry.timestamp).toISOString()
            },
            null,
            2
          )
        )
      } else {
        console.log(`Key "${key}" not found`)
        process.exit(1)
      }
    } finally {
      await engine.dispose()
    }
  }
)
