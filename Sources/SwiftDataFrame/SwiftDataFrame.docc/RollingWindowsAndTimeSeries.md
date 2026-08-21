# Rolling Windows & Time-Series Transformations

Compute sliding window aggregations, exponentially weighted moving averages, and lagged time-series features.

## Overview

Financial and sensor time-series require localized window computations to capture trends, momentum, and volatility over time.

## 1. Rolling Moving Averages & Volatility

```swift
import SwiftDataFrame

let priceDF = try DataFrame(columns: [
    TypedColumn<Double>(name: "price", values: [100.0, 102.0, 101.5, 105.0, 107.0, 106.5, 110.0])
])

// Add a 3-period lagged column for autoregression
let laggedDF = try priceDF.withLaggedColumn(column: "price", by: 1, targetColumn: "price_lag1")
```

## 2. Exponentially Weighted Moving Average (EMA)

$$\text{EMA}_t = \alpha \cdot x_t + (1 - \alpha) \cdot \text{EMA}_{t-1}, \quad \alpha = \frac{2}{N + 1}$$

Accelerated exponential smoothing eliminates high-frequency noise while preserving phase alignment.

## Topics

### Time Series APIs
- ``DataFrame/withLaggedColumn(column:by:targetColumn:)``
