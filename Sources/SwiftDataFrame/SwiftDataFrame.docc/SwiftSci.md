# ``SwiftDataFrame``

High-Performance Columnar Data Tables built on Apache Arrow.

## Overview

`SwiftDataFrame` is a zero-copy, memory-efficient columnar table engine built natively in Swift for Apple Silicon and Swift 6 strict concurrency.

### Key Capabilities

- **Apache Arrow Integration**: Zero-copy memory sharing and Arrow IPC buffer conversions.
- **Relational Operations**: Fast hash-joins (`inner`, `left`, `right`, `outer`), grouping, and aggregations.
- **Reshaping & Filtering**: Pivot, melt, index-based row gathering, and vDSP mask filtering.
- **Streaming I/O**: Direct HTTP/HTTPS dataset streaming via `DataFrame.readURL` and zero-allocation CSV parsing.

### Example Usage

```swift
import SwiftDataFrame

let df = try await DataFrame(csv: fileURL)
let summary = df.filter("Age" > 18.0).groupBy("Pclass").mean()
summary.debugPrint()
```

## Topics

### Core Guides
- <doc:GettingStarted>
- <doc:MachineLearningWorkflow>
- <doc:TimeSeriesForecasting>
