# Time Series Decomposition & Forecasting with SwiftForecast

Perform classical time series decomposition, SARIMA forecasting, and Kalman filtering.

## Overview

`SwiftForecast` provides high-performance vectorized time series tools designed for financial modeling, signal processing, and predictive analytics.

### 1. Classical Decomposition

```swift
import SwiftForecast

let values: [Double] = [...] // Monthly sales data
let decomp = try TimeSeriesDecomposition.decompose(
    values,
    period: 12,
    model: .additive
)

print("Trend:", decomp.trend)
print("Seasonal:", decomp.seasonal)
print("Residuals:", decomp.resid)
```

### 2. Auto-Regressive Integrated Moving Average (ARIMA)

```swift
let arima = ARIMA(p: 1, d: 1, q: 1)
try arima.fit(series: values)

let forecasts = try arima.forecast(steps: 6)
print("6-Month Forecast:", forecasts)
```

### 3. GARCH Volatility Modeling

```swift
let garch = GARCH(p: 1, q: 1)
try garch.fit(returns: stockReturns)
let volatility = garch.conditionalVolatility
```
