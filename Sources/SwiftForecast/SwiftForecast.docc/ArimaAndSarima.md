# ARIMA & SARIMA Time Series Models

Forecast univariate time series using Autoregressive Integrated Moving Average (ARIMA) and Seasonal ARIMA (SARIMA).

## Overview

Model temporal dependencies, trends, and seasonal cycles with statistical forecasting engines.

### 1. ARIMA(p, d, q) Forecasting

```swift
import SwiftForecast

let timeSeries: [Double] = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118]
let arima = ARIMA(p: 1, d: 1, q: 1)
try arima.fit(series: timeSeries)

let forecast = try arima.predict(steps: 6)
print("Forecasted 6 steps: \(forecast)")
```
