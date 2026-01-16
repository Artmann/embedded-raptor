import { command } from 'cleye'
import { EmbeddingEngine } from '../engine'
import { sharedFlags } from './flags'

export const deleteCmd = command(
  {
    name: 'delete',
    parameters: ['<key>'],
    flags: {
      ...sharedFlags
    },
    help: {
      description: 'Delete an embedding entry by key',
      examples: [
        'raptor delete doc1',
        'raptor delete -s ./my-db.raptor mykey'
      ]
    }
  },
  async (argv) => {
    const engine = new EmbeddingEngine({
      storePath: argv.flags.storePath
    })

    try {
      const [key] = argv._
      const deleted = await engine.delete(key)

      if (deleted) {
        console.log(`Deleted key "${key}"`)
      } else {
        console.log(`Key "${key}" not found`)
        process.exit(1)
      }
    } finally {
      await engine.dispose()
    }
  }
)
