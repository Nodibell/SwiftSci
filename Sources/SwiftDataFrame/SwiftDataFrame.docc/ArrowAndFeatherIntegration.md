# Apache Arrow & Feather Zero-Copy Integration

Learn how SwiftDataFrame leverages Apache Arrow memory buffers and Feather IPC format for ultra-fast, zero-copy tabular serialization.

## Overview

`SwiftDataFrame` integrates directly with Apache Arrow, the industry standard for in-memory columnar data. By utilizing Arrow's contiguous memory buffers and bitmapped null representations, SwiftSci achieves seamless interoperability with Python, Rust, and C++ analytical ecosystems without data serialization penalties.

## 1. Arrow Memory Layout & Bitmaps

In traditional row-oriented memory layouts, checking for null values requires conditional branching or sentinel values, causing CPU cache misses. Arrow organizes data into contiguous columnar arrays with an accompanying validity bitmap:

```
┌───────────────────────────────────────────────────────────────┐
│                     TYPED COLUMN (ARROW BUFFER)               │
├───────────────────────────────────────────────────────────────┤
│ Values Buffer : [ 42.0,  18.5,  99.9,  0.0,   12.4,  55.1 ]   │
│ Validity Mask : [  1  ,   1  ,   1  ,   0 ,    1  ,   1   ]   │
│                                        └── NULL               │
└───────────────────────────────────────────────────────────────┘
```

* **Validity Bitmaps**: Each bit represents the presence (1) or absence (0) of a value at that index.
* **SIMD Null Filtering**: Vectorized operations skip null evaluations using SIMD bitmasks (`vDSP_vcmprsD`), allowing full CPU register utilization.

## 2. Feather File Format (Arrow IPC)

Feather is a lightweight, portable binary columnar format built on top of Arrow IPC:

```swift
import SwiftDataFrame

// Write DataFrame directly to Arrow Feather format
let df = try DataFrame(columns: [
    TypedColumn<Double>(name: "signal", values: [1.2, 3.4, 5.6]),
    TypedColumn<String>(name: "label", values: ["A", "B", "C"])
])

let featherURL = URL(fileURLWithPath: "/tmp/data.feather")
try FeatherWriter.write(dataFrame: df, to: featherURL)

// Read Feather file back with zero memory overhead
let loadedDF = try FeatherReader.read(from: featherURL)
print("Loaded \(loadedDF.rowCount) rows and \(loadedDF.columnCount) columns.")
```

## 3. Lazy Feather Streaming

For large datasets, SwiftDataFrame supports streaming chunked reads:

```swift
let lazyDF = try LazyDataFrame.fromFeather(url: featherURL)
let filtered = try lazyDF
    .filter(column: "signal", operation: .greaterThan(2.0))
    .select(["label"])
    .collect()
```

## Topics

### Arrow Classes & Readers
- ``FeatherReader``
- ``FeatherWriter``
- ``TypedColumn``
