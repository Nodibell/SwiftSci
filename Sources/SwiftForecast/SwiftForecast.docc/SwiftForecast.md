# ``SwiftForecast``

Time Series & Volatility Forecasting Models.

## Overview

`SwiftForecast` delivers univariate time series forecasting, ARMA/ARIMA modeling, GARCH volatility, and state-space Kalman filtering.

### Key Capabilities

- **Statistical Models**: `ARIMA(p, d, q)` and Seasonal `SARIMA` for trend and seasonality forecasting.
- **Exponential Smoothing**: Holt-Winters single, double, and triple exponential smoothing.
- **Volatility Modeling**: `GARCHModel` for financial volatility forecasting.
- **Nonlinear & Dynamic Systems**: `KoopmanOperator` for Extended Dynamic Mode Decomposition (EDMD) and spectral analysis.
- **Decomposition & Filtering**: Additive/multiplicative `TimeSeriesDecomposition` and 1D `KalmanFilter`.

### Example Usage

```swift
import SwiftForecast

let arima = ARIMA(p: 1, d: 1, q: 1)
try arima.fit(series: timeSeries)
let forecast = try arima.predict(steps: 6)
```

## Topics

### Guides & Tutorials
- <doc:ArimaAndSarima>
- <doc:SmoothingAndDecomposition>
- <doc:KoopmanOperator>

