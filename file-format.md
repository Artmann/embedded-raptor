# Embedded Raptor Database File Format (v2)

## Overview

This document describes the WAL-based binary file format for the Embedded Raptor
database. The format provides durability through write-ahead logging and
supports insert, update, and delete operations.

## Design Goals

- WAL-based durability (crash-safe)
- Append-only writes for thread safety
- O(1) key lookups via in-memory index
- Support for insert, update, and delete operations
- CRC32 checksums for corruption detection
- Exclusive file locking for multi-process safety

## File Structure

The database consists of three files:

```
database.raptor      # Data file (embeddings)
database.raptor-wal  # Write-ahead log
database.raptor.lock # Exclusive lock file
```

## Write Path

All writes follow this sequence for durability:

```
1. Write data record to .raptor file
2. fsync data file
3. Write WAL entry to .raptor-wal file
4. fsync WAL file (COMMIT POINT)
5. Update in-memory index
```

If a crash occurs before step 4, the operation is not committed. On recovery,
the WAL is replayed to rebuild the index.

## Data File Format (.raptor)

### File Header

**Size:** 16 bytes (fixed)

| Offset | Size | Type   | Description                       |
| ------ | ---- | ------ | --------------------------------- |
| 0      | 4    | bytes  | Magic bytes: "EMBD" (0x454d4244)  |
| 4      | 2    | uint16 | Version number (2 for WAL format) |
| 6      | 4    | uint32 | Embedding dimension (e.g., 384)   |
| 10     | 6    | bytes  | Reserved for future use (zeros)   |

**Endianness:** Little-endian for all multi-byte values (except magic which is
big-endian for ASCII readability)

### Data Record Format

Variable-length records appended after the header:

| Field           | Size        | Type    | Description                             |
| --------------- | ----------- | ------- | --------------------------------------- |
| Magic           | 4 bytes     | uint32  | Record magic: 0xCAFEBABE                |
| Version         | 2 bytes     | uint16  | Record version (2)                      |
| OpType          | 1 byte      | uint8   | Operation: 0=insert, 1=update, 2=delete |
| Flags           | 1 byte      | uint8   | Reserved (0)                            |
| Sequence Number | 8 bytes     | int64   | Monotonic sequence number               |
| Timestamp       | 8 bytes     | int64   | Unix timestamp in milliseconds          |
| Key Length      | 2 bytes     | uint16  | Length of key in bytes                  |
| Key             | variable    | bytes   | UTF-8 encoded key string                |
| Dimension       | 4 bytes     | uint32  | Embedding dimension                     |
| Embedding       | D × 4 bytes | float32 | Embedding vector values                 |
| Checksum        | 4 bytes     | uint32  | CRC32 of all preceding fields           |
| Trailer         | 4 bytes     | uint32  | Record trailer: 0xDEADBEEF              |

**Record Size Calculation:**

```
record_size = 4 + 2 + 1 + 1 + 8 + 8 + 2 + key_length + 4 + (dimension × 4) + 4 + 4
            = 38 + key_length + (dimension × 4)
```

For a 384-dimension embedding with a 10-byte key:

```
38 + 10 + (384 × 4) = 1,584 bytes
```

## WAL File Format (.raptor-wal)

### WAL Entry Format

Fixed 48-byte entries:

| Field           | Offset | Size    | Type   | Description                             |
| --------------- | ------ | ------- | ------ | --------------------------------------- |
| Magic           | 0      | 4 bytes | uint32 | Record magic: 0xCAFEBABE                |
| Version         | 4      | 2 bytes | uint16 | WAL version (1)                         |
| OpType          | 6      | 1 byte  | uint8  | Operation: 0=insert, 1=update, 2=delete |
| Flags           | 7      | 1 byte  | uint8  | Reserved (0)                            |
| Sequence Number | 8      | 8 bytes | int64  | Monotonic sequence number               |
| Offset          | 16     | 8 bytes | uint64 | Byte offset in data file                |
| Length          | 24     | 4 bytes | uint32 | Length of data record                   |
| Key Hash        | 28     | 8 bytes | bytes  | FNV-1a hash of key (truncated)          |
| Reserved        | 36     | 4 bytes | bytes  | Reserved for future use                 |
| Checksum        | 40     | 4 bytes | uint32 | CRC32 of bytes 0-39                     |
| Trailer         | 44     | 4 bytes | uint32 | Record trailer: 0xDEADBEEF              |

## Lock File (.raptor.lock)

A simple lock file created with exclusive flags (`O_CREAT | O_EXCL`). Contains
the PID of the process holding the lock for debugging purposes.

## Operation Types

| Value | Name   | Description                                    |
| ----- | ------ | ---------------------------------------------- |
| 0     | INSERT | New key-value pair                             |
| 1     | UPDATE | Update existing key                            |
| 2     | DELETE | Logical delete (tombstone with zero embedding) |

## Recovery Process

On startup:

1. Acquire exclusive lock
2. Read WAL file sequentially
3. For each valid WAL entry:
   - Validate checksum and trailer
   - Read key from data file at specified offset
   - Update in-memory index with (key → {offset, length, seqNum})
   - For DELETE operations, remove key from index
4. Continue until EOF or corrupted entry (partial write)

## Validation

### Record Validation

1. Check magic bytes match (0xCAFEBABE)
2. Check version is supported
3. Verify checksum matches computed CRC32
4. Check trailer matches (0xDEADBEEF)

### Corruption Handling

- Corrupted records at end of file are ignored (partial write from crash)
- Corrupted records in middle indicate serious corruption
- CRC32 detects single-bit errors and most multi-bit errors

## Space Efficiency

Comparison for 384-dimension embedding with 20-character key:

| Format | Size per Record | Notes                      |
| ------ | --------------- | -------------------------- |
| JSONL  | ~3,200 bytes    | JSON encoding, base64, etc |
| Binary | 1,594 bytes     | 38 + 20 + 1,536            |

**Savings:** ~50% smaller than JSONL

## Concurrency

### Single Writer

- Exclusive lock prevents multiple writers
- All writes are serialized through a mutex
- Write batching improves throughput for bulk inserts

### Multiple Readers (Read-Only Mode)

- Read-only mode skips lock acquisition
- Multiple read-only instances can coexist
- Read-only instances see snapshot at open time
- Cannot write (throws `ReadOnlyError`)

## Version History

| Version | Description                        |
| ------- | ---------------------------------- |
| 1       | Original binary format (no WAL)    |
| 2       | WAL-enabled format with timestamps |

## Implementation Notes

### Thread Safety

- Single writer with exclusive lock
- Multiple concurrent readers in read-only mode
- Write batching groups fsyncs for throughput

### Error Handling

- Always validate magic bytes and checksums
- Handle truncated files gracefully (partial last record)
- Lock acquisition has configurable timeout

### Performance

- In-memory index provides O(1) lookups
- WAL enables crash recovery without full file scan
- Write batching reduces fsync overhead
- Embedding cache speeds up repeated searches
