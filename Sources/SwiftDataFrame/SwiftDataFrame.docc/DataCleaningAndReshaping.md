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

## 3. Deduplication with `DataFrame.unique` and `TypedColumn.typedUnique`

`DataFrame.unique` and `TypedColumn.typedUnique` deduplicate using canonical typed keys (`CanonicalUniqueKey`) rather than string interpolation:

- All floating-point `NaN` payloads canonicalize to a single distinct `NaN` value, separate from `nil` (missing).
- `-0.0` and `+0.0` canonicalize together.
- Typed `Int64`, `Double`, `String`, `Bool`, `Date`, and `nil` cells never collide with string `"nil"` or separator substrings.

```swift
// Remove identical duplicate rows across all columns
let uniqueDF = df.unique

// Deduplicate a single strongly typed column preserving first appearance
let uniqueStations = (df[column: "station"] as? TypedColumn<String>)?.typedUnique
```

## 4. CSV Serialization & Quoted Null-Token Round-Trip

Under `CSVReadOptions.default`, unquoted configured null tokens (`"NA"`, `"null"`, `"NaN"`, `"None"`, and empty unquoted fields) decode as `nil`. To prevent data corruption across write/read cycles:

- `writeCSV(to:)` automatically quotes string values that are empty (`""`) or match an active null token after whitespace trimming.
- `DataFrame(csv:options:)` and streaming readers preserve quoted empty strings and quoted null tokens as literal `String` values across eager, streaming, and chunked paths, while interpreting unquoted tokens and unquoted empty fields as missing (`nil`).

```swift
let original = try DataFrame(columns: [
    TypedColumn<String>(name: "token", values: ["text", "NA", "null", "NaN", "", nil, "None"])
])
try await original.writeCSV(to: csvURL)
let reloaded = try await DataFrame(csv: csvURL)
// reloaded["token"] preserves all literal strings and restores only index 5 as nil.
```

## Topics

### Reshaping APIs
- ``DataFrame/pivot(index:columns:values:aggFunc:)``
- ``DataFrame/melt(idVars:valueVars:varName:valueName:)``
- ``DataFrame/unique``
- ``TypedColumn/typedUnique``
- ``DataFrame/writeCSV(to:)``
- ``DataFrame/init(csv:options:)``
