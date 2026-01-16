import { command } from 'cleye'
import { Wal } from '../storage-engine/wal'
import { opType, fileExtensions } from '../storage-engine/constants'
import { sharedFlags } from './flags'

const opTypeNames: Record<number, string> = {
  [opType.insert]: 'INSERT',
  [opType.update]: 'UPDATE',
  [opType.delete]: 'DELETE'
}

function formatKeyHash(hash: Uint8Array): string {
  return Array.from(hash.slice(0, 8))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('')
}

export const walCmd = command(
  {
    name: 'wal',
    flags: {
      ...sharedFlags
    },
    help: {
      description: 'Display WAL entries in human-readable format',
      examples: ['raptor wal', 'raptor wal -s ./my-db.raptor']
    }
  },
  async (argv) => {
    // Strip .raptor extension if present, then add .raptor-wal
    const basePath = argv.flags.storePath.replace(/\.raptor$/, '')
    const walPath = basePath + fileExtensions.wal
    const wal = new Wal(walPath)

    try {
      let count = 0
      for await (const entry of wal.recover()) {
        const opName = opTypeNames[entry.opType] ?? `UNKNOWN(${entry.opType})`
        const keyHashHex = formatKeyHash(entry.keyHash)

        console.log(`[${entry.sequenceNumber}] ${opName}`)
        console.log(`  offset: ${entry.offset}, length: ${entry.length}`)
        console.log(`  keyHash: ${keyHashHex}`)
        console.log()

        count++
      }

      if (count === 0) {
        console.log('WAL is empty')
      } else {
        console.log(`Total: ${count} entries`)
      }
    } finally {
      await wal.close()
    }
  }
)
