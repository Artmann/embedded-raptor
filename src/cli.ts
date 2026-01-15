#!/usr/bin/env node
import { cli } from 'cleye'
import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

import { store, get, search, deleteCmd } from './commands'
import type { PackageJson } from './types'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

function main(): void {
  const packageJson = JSON.parse(
    readFileSync(resolve(__dirname, '../package.json'), 'utf-8')
  ) as PackageJson

  cli(
    {
      name: 'raptor',
      version: packageJson.version,
      description:
        'An embedding database CLI for storing and searching text indexes',
      commands: [store, get, search, deleteCmd]
    },
    () => {
      // When using commands with handlers, this callback is only called when no command is matched
      // The help text will be shown automatically
    }
  )
}

main()
