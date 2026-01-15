import { command } from 'cleye'
import { EmbeddingEngine } from '../engine'
import { sharedFlags } from './flags'

export const store = command(
  {
    name: 'store',
    description: 'Store a text embedding with a key',
    parameters: ['<key>', '<text>'],
    flags: {
      ...sharedFlags
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
