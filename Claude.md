# Claude.md - Embedded Raptor Project Guide

## Project Overview

**Embedded Raptor** is a lightweight embedding database built for the Bun
runtime that stores text embeddings in an append-only JSONL file format. It
provides semantic search capabilities using cosine similarity with the
BGE-Base-EN embedding model (768 dimensions).

**Key Features:**

- Simple key-value storage for text embeddings
- Semantic search using cosine similarity
- Append-only JSONL storage format
- Both CLI and programmatic API
- Memory-efficient chunked file reading
- Full TypeScript support

**Tech Stack:**

- Runtime: Bun
- Language: TypeScript
- Embedding Model: FastEmbed (BAAI/bge-base-en)
- Build Tool: Rolldown
- Test Framework: Vitest

## Architecture

### Core Components

#### 1. EmbeddingEngine (`src/engine.ts`)

The main class that orchestrates all operations:

- `store(key, text)` - Stores text with auto-generated embedding
- `get(key)` - Retrieves entry by key (most recent if duplicates)
- `search(query, limit, minSimilarity)` - Performs semantic search
- `generateEmbedding(text)` - Generates embeddings using FastEmbed
- `cosineSimilarity(a, b)` - Calculates cosine similarity between vectors

#### 2. EmbeddingFileReader (`src/embedding-file-reader.ts`)

Memory-efficient file reader that:

- Reads JSONL files in reverse order (most recent entries first)
- Processes file in 64KB chunks to handle large files
- Automatically deduplicates by key (newest entry wins)
- Uses async generator pattern for streaming

#### 3. CandidateSet (`src/candidate-set.ts`)

Optimized data structure for top-N search results:

- Maintains fixed-size priority queue of highest similarity matches
- Early rejection of candidates below minimum threshold
- Efficient O(n) insertion for bounded size
- Sorts results by similarity (descending)

#### 4. CLI (`src/cli.ts`)

Command-line interface with three commands:

- `raptor store <key> <text>` - Store new embedding
- `raptor get <key>` - Retrieve by key
- `raptor search <query>` - Semantic search

### Data Flow

**Store Operation:**

```
User Input → Generate Embedding → Append to JSONL → Success
```

**Search Operation:**

```
Query → Generate Embedding → Read File (Chunked) →
Calculate Similarities → CandidateSet (Top-N) → Return Results
```

**Get Operation:**

```
Key → Read File (Chunked) → Find First Match → Return Entry
```

## File Structure

```
src/
├── commands/           # CLI command implementations
│   ├── flags.ts       # Shared CLI flags (--storePath)
│   ├── get.ts         # Get command
│   ├── search.ts      # Search command
│   └── store.ts       # Store command
├── candidate-set.ts   # Top-N search results data structure
├── cli.ts             # CLI entry point
├── embedding-file-reader.ts  # Efficient file reading
├── engine.ts          # Core EmbeddingEngine class
├── index.ts           # Library export
└── types.ts           # TypeScript definitions
```

## Development Workflows

### Setup

```bash
bun install
```

### Development

```bash
bun run dev          # Run main entry point
bun run cli          # Run CLI in development mode
```

### Code Quality

```bash
bun run typecheck    # TypeScript type checking
bun run lint         # Run ESLint
bun run lint:fix     # Auto-fix linting issues
bun run format       # Format code with Prettier
bun run format:check # Check formatting
```

### Testing

```bash
bun run test         # Run tests once
bun run test:watch   # Run tests in watch mode
bun run test:ui      # Run tests with UI
```

### Build

```bash
bun run build        # Build for distribution (ESM + CJS + types)
```

## Common Development Tasks

### Adding a New Command

1. Create command file in `src/commands/`
2. Import shared flags from `src/commands/flags.ts`
3. Use `EmbeddingEngine` for operations
4. Export from `src/commands/index.ts`
5. Register in `src/cli.ts`

### Modifying Storage Format

- Update `EmbeddingEntry` type in `src/types.ts`
- Modify `store()` method in `src/engine.ts`
- Update `EmbeddingFileReader` parsing in `src/embedding-file-reader.ts`
- Add migration logic if needed

### Changing Embedding Model

- Update FastEmbed initialization in `src/engine.ts`
- Update dimension references (currently 768 for BGE-Base-EN)
- Clear `local_cache/` to download new model
- Update README with new model details

### Adding New Search Filters

- Extend `SearchOptions` in `src/types.ts`
- Modify `search()` method in `src/engine.ts`
- Update `EmbeddingFileReader` if needed for filtering
- Add CLI flags in `src/commands/search.ts`

## Code Style and Conventions

### TypeScript Rules

- **Strict Mode**: All strict TypeScript checks enabled
- **No `any`**: Use proper types or `unknown`
- **No Non-null Assertions**: Avoid `!` operator
- **Prefer Nullish Coalescing**: Use `??` over `||`
- **No Floating Promises**: Always await or handle promises
- **No Unused Vars**: Prefix with `_` if intentionally unused

### Formatting (Prettier)

- Single quotes
- No semicolons
- 2-space indentation
- Trailing commas: none
- Arrow function parens: always

### File Naming

- Kebab-case for files: `embedding-file-reader.ts`
- Test files: `*.test.ts`
- PascalCase for classes: `EmbeddingEngine`, `CandidateSet`
- camelCase for functions and variables

### Import Organization

1. External dependencies
2. Internal modules
3. Types (using `import type`)

## Testing Guidelines

### Test Files

Each component has a corresponding `.test.ts` file:

- `engine.test.ts` - Core engine functionality
- `candidate-set.test.ts` - CandidateSet behavior
- `embedding-file-reader.test.ts` - File reading logic

### Testing Patterns

- Use `describe()` blocks to group related tests
- Use `it()` for individual test cases
- Clean up test data in `afterEach()` or `afterAll()`
- Use `expect()` assertions from Vitest
- Mock file system operations when needed
- Test edge cases: empty files, duplicates, large datasets

### Running Specific Tests

```bash
bun test src/engine.test.ts          # Run single file
bun test -t "search returns results" # Run by pattern
```

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on:

- Push to `main` or `develop` branches
- Pull requests to `main`

**Jobs:**

1. **install** - Cache and install dependencies
2. **typecheck** - TypeScript type checking
3. **lint** - ESLint with zero warnings (`lint:check`)
4. **format** - Prettier formatting check
5. **test** - Run test suite
6. **build** - Build package and upload artifacts

All jobs must pass for CI to succeed.

## Storage Format

### JSONL Structure

Each line is a JSON object:

```json
{"key":"doc1","text":"original text","embedding":[0.1,0.2,...],"timestamp":1234567890}
```

### Important Notes

- **Append-only**: New entries are always appended
- **Deduplication**: Most recent entry for a key wins
- **Default Path**: `./database.jsonl`
- **Embedding Dimensions**: 768 (BGE-Base-EN model)
- **No Deletion**: To "delete", append new entry with same key

## API Reference

### Programmatic Usage

```typescript
import { EmbeddingEngine } from 'embedded-raptor'

const engine = new EmbeddingEngine({
  storePath: './my-database.jsonl'
})

// Store
await engine.store('doc1', 'Machine learning is awesome')

// Search
const results = await engine.search('AI and ML', 5, 0.7)
// Returns: Array<{ key: string, similarity: number }>

// Get
const entry = await engine.get('doc1')
// Returns: { key, text, embedding, timestamp } | null
```

### CLI Usage

```bash
# Store
raptor store doc1 "Machine learning is awesome"

# Search (max 5 results, min 0.7 similarity)
raptor search "AI and ML" --limit 5 --minSimilarity 0.7

# Get
raptor get doc1

# Custom database path
raptor store doc1 "text" --storePath ./custom.jsonl
```

## Performance Considerations

### Memory Efficiency

- **Chunked Reading**: Reads 64KB chunks instead of entire file
- **Streaming**: Uses async generators to avoid loading all entries
- **Early Termination**: Stops reading when enough results found

### Optimization Opportunities

- **Index Creation**: Add in-memory index for faster lookups
- **Batch Operations**: Support bulk store operations
- **Compression**: Compress JSONL file for storage savings
- **Caching**: Cache frequently accessed embeddings

### Scalability Limits

- JSONL format is linear scan O(n)
- Search performance degrades with file size
- Consider migration to indexed format for >100K entries

## Troubleshooting

### Common Issues

**"Model not found" error:**

- FastEmbed downloads model on first use to `local_cache/`
- Ensure internet connectivity on first run
- Check disk space for ~200MB model

**Slow searches:**

- JSONL scans entire file for search
- Consider reducing file size or adding indexes
- Use higher `minSimilarity` threshold to reduce candidates

**Out of memory:**

- Check file size - may need chunking optimization
- Reduce batch size in `EmbeddingFileReader`
- Consider splitting database into multiple files

**Type errors:**

- Run `bun run typecheck` to see all errors
- Ensure strict mode compliance
- Check `tsconfig.json` for configuration

## Platform-Specific Notes

### Bun on Windows

**File open with numeric constants bug:**

Bun on Windows has a bug where `fs.open()` fails with `ENOENT` when using
numeric constants (`constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY`),
even when the parent directory exists. The workaround is to use string flags
instead:

```typescript
// Broken on Bun/Windows:
import { constants } from 'node:fs'
await open(path, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY)

// Works everywhere:
await open(path, 'wx') // 'wx' = O_CREAT | O_EXCL | O_WRONLY
```

This affects the `FileLock` class in `src/storage-engine/file-lock.ts`.

## Additional Resources

- **CODE_STYLE.md** - Detailed code style guide and conventions
- **README.md** - User-facing documentation
- **package.json** - Scripts and dependencies
- **tsconfig.json** - TypeScript configuration
- **.eslintrc.json** - Linting rules
- **rolldown.config.ts** - Build configuration
- **vitest.config.ts** - Test configuration

## Quick Commands Reference

| Task                 | Command                                                                       |
| -------------------- | ----------------------------------------------------------------------------- |
| Install dependencies | `bun install`                                                                 |
| Run tests            | `bun test`                                                                    |
| Type check           | `bun run typecheck`                                                           |
| Lint code            | `bun run lint`                                                                |
| Format code          | `bun run format`                                                              |
| Build package        | `bun run build`                                                               |
| Run CLI              | `bun run cli`                                                                 |
| Run all checks       | `bun run typecheck && bun run lint:check && bun run format:check && bun test` |
