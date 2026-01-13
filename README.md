# Embedded Raptor

[![CI](https://img.shields.io/github/actions/workflow/status/artmann/embedded-raptor/ci.yml?branch=main&label=CI&logo=github)](https://github.com/artmann/embedded-raptor/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/embedded-raptor.svg?logo=npm)](https://www.npmjs.com/package/embedded-raptor)
[![npm downloads](https://img.shields.io/npm/dm/embedded-raptor.svg)](https://www.npmjs.com/package/embedded-raptor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg?logo=typescript)](https://www.typescriptlang.org/)

> A lightweight semantic search database with text embeddings for Node.js and
> Bun

Embedded Raptor lets you build semantic search into your applications with just
a few lines of code. Store text, search by meaning, and find similar
content—perfect for RAG systems, chatbots, and recommendation engines.

## What is Embedded Raptor?

Embedded Raptor is an embedding database that automatically converts text into
vector embeddings and stores them in an efficient binary format. Instead of
searching by exact keywords, you can search by semantic similarity—finding
documents that mean the same thing, even if they use different words.

**Example:** Search for "how to reset password" and find results like "forgot my
login credentials" or "change account password".

## Why Embedded Raptor?

- **Simple API** - No complex setup, just store and search
- **Semantic Search** - Find content by meaning, not just keywords
- **Fast & Efficient** - Binary storage format ~50% smaller than JSON
- **Zero Dependencies** - Embeddings generated locally, no API keys needed
- **Works Everywhere** - Compatible with Node.js and Bun
- **Built for RAG** - Perfect for Retrieval Augmented Generation systems

## Use Cases

- **FAQ Bots** - Match user questions to answers by meaning
- **Document Search** - Semantic search over documentation or knowledge bases
- **Code Search** - Find code snippets by describing what they do
- **Content Recommendations** - "More like this" functionality
- **RAG Systems** - Retrieve relevant context for LLM prompts

## Installation

```bash
# Using npm
npm install embedded-raptor

# Using bun
bun add embedded-raptor
```

## Quick Start

### Programmatic API

```typescript
import { EmbeddingEngine } from 'embedded-raptor'

const engine = new EmbeddingEngine({
  storePath: './my-database.raptor'
})

// Store documents
await engine.storeMany([
  { key: 'doc1', text: 'How to reset your password' },
  { key: 'doc2', text: 'Machine learning basics' },
  { key: 'doc3', text: 'Getting started with Bun' }
])

// Search by meaning
const results = await engine.search('forgot my password', 5)
console.log(results[0].key) // 'doc1' - matched by meaning!
console.log(results[0].similarity) // 0.87 - high similarity score
```

### Command Line Interface

```bash
# Store documents
raptor store doc1 "How to reset your password"
raptor store doc2 "Machine learning basics"

# Search by meaning
raptor search "forgot my password" --limit 5

# Retrieve by key
raptor get doc1
```

## Examples

See the [examples/](examples/) directory for complete, runnable examples:

| Example                    | Description                                             | Run                                             |
| -------------------------- | ------------------------------------------------------- | ----------------------------------------------- |
| **Document Search / RAG**  | Semantic search over documentation chunks               | `bun run examples/01-document-search.ts`        |
| **FAQ Bot**                | Match user questions to FAQs with confidence thresholds | `bun run examples/02-faq-bot.ts`                |
| **Code Snippet Library**   | Search code by natural language descriptions            | `bun run examples/03-code-snippets.ts`          |
| **Content Recommendation** | "More like this" functionality                          | `bun run examples/04-content-recommendation.ts` |

## API Reference

### `EmbeddingEngine`

#### `constructor(options)`

Create a new embedding engine.

```typescript
const engine = new EmbeddingEngine({
  storePath: './database.raptor' // Path to storage file
})
```

#### `store(key, text)`

Store a single text entry with auto-generated embedding.

```typescript
await engine.store('doc1', 'The quick brown fox')
```

#### `storeMany(items)`

Store multiple entries in batch (faster than multiple `store()` calls).

```typescript
await engine.storeMany([
  { key: 'doc1', text: 'First document' },
  { key: 'doc2', text: 'Second document' }
])
```

#### `search(query, limit?, minSimilarity?)`

Search for semantically similar entries.

```typescript
const results = await engine.search(
  'artificial intelligence', // query text
  10, // max results (default: 10)
  0.7 // min similarity 0-1 (default: 0)
)

// Results are sorted by similarity (highest first)
results.forEach((result) => {
  console.log(result.key) // Document key
  console.log(result.similarity) // Similarity score 0-1
})
```

#### `get(key)`

Retrieve a specific entry by key.

```typescript
const entry = await engine.get('doc1')
if (entry) {
  console.log(entry.key) // 'doc1'
  console.log(entry.embedding) // [0.1, 0.2, ...] (768 dimensions)
}
```

## CLI Reference

### Commands

```bash
# Store text
raptor store <key> <text> [--storePath path]

# Search for similar text
raptor search <query> [--limit 10] [--minSimilarity 0] [--storePath path]

# Get by key
raptor get <key> [--storePath path]
```

### Options

- `-s, --storePath` - Path to database file (default: `./database.raptor`)
- `-l, --limit` - Maximum results to return (default: 10)
- `-m, --minSimilarity` - Minimum similarity threshold 0-1 (default: 0)

### Examples

```bash
# Store documents
raptor store doc1 "The quick brown fox jumps over the lazy dog"
raptor store doc2 "Machine learning is a subset of artificial intelligence"
raptor store doc3 "Bun is a fast JavaScript runtime"

# Search with default settings
raptor search "artificial intelligence"

# Search with custom limit and threshold
raptor search "AI and ML" --limit 3 --minSimilarity 0.7

# Use custom database path
raptor store key1 "Some text" --storePath ./data/custom.raptor
```

## How It Works

1. **Text → Embeddings**: Embedded Raptor uses the BGE-Base-EN model to convert
   text into 768-dimensional vector embeddings
2. **Storage**: Embeddings are stored in an efficient binary format (.raptor
   files)
3. **Search**: When you search, Embedded Raptor compares your query embedding
   against all stored embeddings using cosine similarity
4. **Results**: Returns the most similar results ranked by similarity score

**Embedding Model**:
[BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en-v1.5) (768
dimensions)

## Performance

- **Batch operations**: Use `storeMany()` for faster bulk inserts
- **Memory efficient**: Reads file in 64KB chunks, handles large databases
- **Fast search**: Cosine similarity comparison across all embeddings
- **Deduplication**: Latest entry automatically used for duplicate keys

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
development setup, architecture details, and guidelines.

## License

MIT © [Christoffer Artmann](mailto:artgaard@gmail.com)
