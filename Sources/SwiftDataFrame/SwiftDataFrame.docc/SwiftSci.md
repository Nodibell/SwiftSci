# ``SwiftDataFrame``

High-Performance Columnar Data Tables built on Apache Arrow.

## Overview

`SwiftDataFrame` is a zero-copy, memory-efficient columnar table engine built natively in Swift for Apple Silicon and Swift 6 strict concurrency.

### Key Capabilities

- **Apache Arrow & Parquet Integration**: Zero-copy memory sharing, Arrow IPC buffer conversions, and pure-Swift Apache Parquet reader/writer (`ParquetReader`/`ParquetWriter`) with Snappy decompression.
- **Out-of-Core Data Streaming**: `ChunkedDataFrame` streaming pipeline with zero-copy POSIX `MemoryMappedReader` partitioning.
- **Relational Operations & SIMD Joins**: SIMD-accelerated typed hash-joins (`inner`, `left`, `right`, `outer`), grouping, and aggregations.
- **Reshaping & Filtering**: Pivot, melt, index-based row gathering, and vDSP mask filtering.
- **Deduplication**: `DataFrame.unique`, `AnyColumn.unique`, and `TypedColumn.unique` with first-occurrence order preservation.
- **Streaming I/O**: Direct HTTP/HTTPS dataset streaming via `DataFrame.readURL` and zero-allocation CSV parsing.

### Example Usage

```swift
import SwiftDataFrame

// Read Parquet / CSV / Chunked Streams
let df = try await DataFrame(csv: fileURL)
let parquetData = try ParquetWriter.write(df)
let loadedDF = try ParquetReader.read(from: parquetData)
```

## Topics

### Core Guides
- <doc:GettingStarted>
- <doc:MachineLearningWorkflow>
- <doc:TimeSeriesForecasting>
