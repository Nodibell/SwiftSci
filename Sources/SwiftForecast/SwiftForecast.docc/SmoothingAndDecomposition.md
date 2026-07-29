# Exponential Smoothing & Decomposition

Decompose time series into trend, seasonality, and residuals, and apply Holt-Winters exponential smoothing.

## Overview

Extract seasonal components and model volatility using GARCH and Kalman Filtering.

### 1. Seasonal Additive Decomposition

```swift
import SwiftForecast

let decomp = TimeSeriesDecomposition.decompose(series: timeSeries, period: 4, type: .additive)
print("Trend: \(decomp.trend)")
print("Seasonal: \(decomp.seasonal)")
```
