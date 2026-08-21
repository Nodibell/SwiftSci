# Data Cleaning, Pivoting & Reshaping

Transform messy tabular data using type-safe null handling, column casting, pivoting, and deduplication.

## Overview

Real-world datasets require extensive preprocessing before model training. `SwiftDataFrame` provides declarative, vector-accelerated functions to clean missing data, cast column types, reshape wide/long representations, and remove duplicates.

## 1. Handling Missing Data (Null Imputation)

```swift
import SwiftDataFrame

var df = try DataFrame(columns: [
    TypedColumn<Double>(name: "temperature", values: [21.5, nil, 23.0, nil, 25.5]),
    TypedColumn<String>(name: "station", values: ["A", "B", "A", nil, "B"])
])

// Fill null values with a constant or column mean
let filledDF = try df
    .fillNulls(column: "temperature", with: 20.0)
    .fillNulls(column: "station", with: "Unknown")

// Drop rows with any null values
let cleanDF = try df.dropNulls()
```

## 2. Long-to-Wide Pivoting & Wide-to-Long Melting

```swift
// Pivot long format table to wide format
let wideDF = try df.pivot(
    index: "station",
    columns: "date",
    values: "temperature",
    aggregate: .mean
)

// Melt wide format back to long format
let longDF = try wideDF.melt(
    idVariables: ["station"],
    valueVariables: ["2026-01-01", "2026-01-02"],
    variableColumnName: "date",
    valueColumnName: "temperature"
)
```

## 3. Deduplication with `DataFrame.unique`

```swift
// Remove identical duplicate rows across all columns
let uniqueDF = df.unique
```

## Topics

### Reshaping APIs
- ``DataFrame/pivot(index:columns:values:aggregate:)``
- ``DataFrame/melt(idVariables:valueVariables:variableColumnName:valueColumnName:)``
- ``DataFrame/unique``
