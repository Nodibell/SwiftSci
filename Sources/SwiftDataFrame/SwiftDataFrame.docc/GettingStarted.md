# Getting Started with SwiftSci DataFrame

Learn how to load CSV data, manipulate columns, filter rows, perform group-by aggregations, and join datasets.

## Overview

`SwiftDataFrame` is a high-performance, column-oriented data framework built natively in Swift for Apple Silicon and modern Swift 6 strict concurrency.

### 1. Loading CSV Data

```swift
import Foundation
import SwiftDataFrame

let fileURL = URL(filePath: "/path/to/data.csv")
let df = try await DataFrame(csv: fileURL)

print("Shape: \(df.shape.rows) rows × \(df.shape.columns) columns")
df.head(5).debugPrint()
```

### 2. Feature Engineering

```swift
let engineered = try df
    .addColumn("AgeImputed", as: Double.self) { row in
        row.value(column: "Age", as: Double.self) ?? 29.0
    }
    .addColumn("IsAlone", as: Double.self) { row in
        let sibSp = row.value(column: "SibSp", as: Int64.self) ?? 0
        let parch = row.value(column: "Parch", as: Int64.self) ?? 0
        return (sibSp + parch) == 0 ? 1.0 : 0.0
    }
```

### 3. Aggregations & Joins

```swift
let summary = df.groupBy("Pclass").mean()
summary.debugPrint()

let joined = try df1.join(df2, on: "id", how: .inner)
```
