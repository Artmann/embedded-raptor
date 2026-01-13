import { defineConfig } from 'rolldown'

export default defineConfig({
  input: {
    index: 'src/index.ts',
    cli: 'src/cli.ts'
  },
  output: [
    {
      dir: 'dist',
      format: 'esm',
      entryFileNames: '[name].mjs',
      chunkFileNames: '[name]-[hash].mjs'
    },
    {
      dir: 'dist',
      format: 'cjs',
      entryFileNames: '[name].cjs',
      chunkFileNames: '[name]-[hash].cjs'
    }
  ],
  external: [
    'node:fs/promises',
    'node:path',
    'node:fs',
    'node:url',
    'fs/promises',
    'path',
    'fs',
    'url',
    'cleye',
    'fastembed',
    'tiny-invariant',
    '@xenova/transformers',
    'node-llama-cpp'
  ]
})
