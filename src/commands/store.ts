import { command } from 'cleye'
import { EmbeddingEngine } from '../engine'
import { sharedFlags } from './flags'

export const store = command(
  {
    name: 'store',
    parameters: ['<key>', '<text>'],
    flags: {
      ...sharedFlags
    },
    help: {
      description: 'Store a text embedding with a key',
      examples: [
        'raptor store doc1 "Machine learning is awesome"',
        'raptor store -s ./my-db.raptor mykey "Some text to embed"'
      ]
    }
  },
  async (argv) => {
    const engine = new EmbeddingEngine({
      storePath: argv.flags.storePath
    })

    try {
      const [key, text] = argv._
      await engine.store(key, text)
      console.log(`Stored embedding for key: ${key}`)
    } finally {
      await engine.dispose()
    }
  }
)
