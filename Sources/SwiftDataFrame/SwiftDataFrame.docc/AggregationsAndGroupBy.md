# High-Performance GroupBy & Vectorized Aggregations

Master high-throughput columnar grouping, multi-key aggregations, and window transformations in SwiftDataFrame.

## Overview

Tabular analytical queries frequently demand grouping by categorical columns and computing summary statistics (mean, sum, variance, min, max, count) across numerical dimensions. `SwiftDataFrame` provides SIMD-accelerated grouping algorithms optimized for Apple Silicon unified memory.

## 1. Basic GroupBy Aggregation

```swift
import SwiftDataFrame

let df = try DataFrame(columns: [
    TypedColumn<String>(name: "category", values: ["Tech", "Finance", "Tech", "Healthcare", "Finance"]),
    TypedColumn<Double>(name: "revenue", values: [120.5, 340.0, 210.0, 95.0, 450.5]),
    TypedColumn<Double>(name: "growth", values: [0.15, 0.08, 0.22, 0.05, 0.12])
])

// Compute mean revenue and sum of growth grouped by category
let grouped = try df.groupBy("category")
    .aggregate([
        "revenue": .mean,
        "growth": .sum
    ])

grouped.debugPrint()
```

## 2. Multi-Column Grouping

You can group by multiple categorical dimensions simultaneously:

```swift
let multiGrouped = try df.groupBy(["region", "product_tier"])
    .aggregate([
        "sales": .sum,
        "discount": .mean,
        "quantity": .count
    ])
```

## 3. Performance & Memory Efficiency

* **Single-Pass Hash Accumulation**: Group indices are computed using contiguous hash tables, avoiding unnecessary row permutations.
* **vDSP Vector Reductions**: Numerical summaries within each group are calculated using Apple Accelerate SIMD routines (`vDSP_sveD`, `vDSP_meanvD`).

## Topics

### Grouping APIs
- ``DataFrame/groupBy(_:)``
