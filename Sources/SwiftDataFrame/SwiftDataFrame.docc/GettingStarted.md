# Getting Started with SwiftSci DataFrame

A complete guide to columnar data loading, $O(1)$ indexing, zero-allocation row iteration, group-by aggregations, joins, and machine learning exports.

## Overview

`SwiftDataFrame` is an immutable, columnar data analysis engine built natively in Swift 6 for Apple Silicon. It combines Apache Arrow / Feather zero-copy memory buffers, $O(1)$ schema lookups, and SIMD bitmask filtering.

```
DataFrame (Value Type: Immutable Struct + Copy-On-Write)
┌─────────────────────────────────────────────────────────────┐
│ Schema (FieldMap: O(1) dictionary lookup by column name)    │
├─────────────────────────────────────────────────────────────┤
│ Column 0: TypedColumn<Int64>   [ 101, 102, 103, 104 ]       │
│ Column 1: TypedColumn<String>  [ "A", "B", "A", nil ]       │
│ Column 2: TypedColumn<Double>  [ 12.5, 45.0, 33.2, 98.1 ]   │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Loading Datasets (CSV, Feather & Parquet)

Load data from local files or memory buffers using streaming asynchronous APIs:

```swift
import Foundation
import SwiftDataFrame

// 1. Load CSV with automatic delimiter and type inference
let csvURL = URL(fileURLWithPath: "passengers.csv")
let df = try await DataFrame(csv: csvURL)

print("Loaded DataFrame: \(df.shape.rows) rows x \(df.shape.columns) columns")
df.head(5).debugPrint()

// 2. Load binary Feather / Apache Arrow IPC format (Zero-Copy)
let featherURL = URL(fileURLWithPath: "dataset.feather")
let featherDF = try await DataFrame(feather: featherURL)

// 3. Load compressed Apache Parquet format
let parquetURL = URL(fileURLWithPath: "telemetry.parquet")
let parquetDF = try await DataFrame(parquet: parquetURL)
```

---

## 2. $O(1)$ Column Access & Zero-Allocation Row Iteration

Access typed columns by name or iterate through rows using `DataFrameRowSequence`:

```swift
// 1. O(1) Typed Column Subscript
if let ageCol = df[column: "Age", as: Double.self] {
    let meanAge = ageCol.mean
    print("Mean Age: \(meanAge ?? 0.0)")
}

// 2. Zero-Allocation Row Iteration (Bypasses heap dictionary creation)
for row in df.rows {
    if let city = row.string("City"), let score = row.double("Score") {
        if score > 90.0 {
            print("Top Performer in \(city): Score = \(score)")
        }
    }
}
```

---

## 3. High-Performance Filtering & Feature Transformation

Filter datasets using SIMD bitmasks or transform columns functional-style:

```swift
// 1. Fast condition-based filtering
let filtered = try df.filter(column: "Fare", where: .greaterThan(50.0))

// 2. Predicate filter closure
let adultFirstClass = df.filter { row in
    (row.int("Pclass") == 1) && ((row.double("Age") ?? 0) >= 18.0)
}

// 3. Add calculated feature column
let engineeredDF = try df.addColumn("FarePerYear", as: Double.self) { row in
    guard let fare = row.double("Fare"), let age = row.double("Age"), age > 0 else { return nil }
    return fare / age
}
```

---

## 4. GroupBy Aggregations & Joins

Group by categorical columns and aggregate numeric columns:

```swift
// 1. GroupBy with Mean Aggregation
let deptAverages = df.groupBy("Department").mean()
deptAverages.debugPrint()

// 2. High-Performance SIMD Hash Join
let employees = try DataFrame(columns: [
    TypedColumn<Int64>(name: "id", values: [1, 2, 3]),
    TypedColumn<String>(name: "name", values: ["Alice", "Bob", "Charlie"])
])
let departments = try DataFrame(columns: [
    TypedColumn<Int64>(name: "id", values: [1, 2, 3]),
    TypedColumn<String>(name: "dept", values: ["AI", "Cloud", "Security"])
])

let merged = try employees.join(departments, on: "id", how: .inner)
merged.debugPrint()
```

---

## 5. Exporting Flat 1D Matrices for ML

Export tabular data directly into contiguous flat buffers for machine learning training:

```swift
let (flatMatrix, rows, cols) = try df.toFlatFeatureMatrix(["Age", "Fare", "Pclass"])
print("Generated flat feature matrix: \(rows) samples x \(cols) features.")
```
