# Concurrent Order Selection with AutoARIMA

Discover optimal autoregressive, integrated, and moving average orders across \((p, d, q) \times (P, D, Q)_s\) grids using parallel Swift Concurrency.

## Overview

Manually diagnosing autocorrelation functions (ACF) and partial autocorrelation functions (PACF) to select ARIMA orders is time-consuming and error-prone. Standard automated packages like R's `forecast::auto.arima` or Python's `pmdarima` search across candidate orders sequentially.

`AutoARIMA` evaluates the entire hyperparameter lattice simultaneously across all CPU cores on Apple Silicon using structured Swift Concurrency (`withThrowingTaskGroup`).

## Grid Search Architecture

```
           Observed Time Series [y_1, y_2, ..., y_N]
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
ARIMA(1, 1, 0)          ARIMA(2, 1, 1)          ARIMA(0, 1, 2)
Task 1                  Task 2                  Task 3
   │                       │                       │
   ├─ Hannan-Rissanen      ├─ Hannan-Rissanen      ├─ Hannan-Rissanen
   └─ Compute AIC/BIC      └─ Compute AIC/BIC      └─ Compute AIC/BIC
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               ▼
                withThrowingTaskGroup Collector
                               │
                               ▼
                 Sorted Leaderboard & Best Model
```

### Information Criteria

- **AIC (Akaike Information Criterion)**: Penalizes model complexity while rewarding likelihood:
  $$\text{AIC} = 2k - 2\ln(L)$$
- **BIC (Bayesian Information Criterion)**: Imposes a stronger penalty proportional to sample size:
  $$\text{BIC} = k\ln(N) - 2\ln(L)$$

## Example Usage

```swift
import SwiftForecast

// 1. Initialize AutoARIMA with bounded search ranges
let autoArima = AutoARIMA(
    maxP: 3,
    maxD: 2,
    maxQ: 3,
    criterion: .aic
)

// 2. Concurrently fit candidates across all cores
let result = try await autoArima.fit(series: historicalPrices)
print("Optimal Order: \(autoArima.bestOrder!) with AIC: \(autoArima.bestScore!)")

// 3. Generate multi-step forecasts
let forecast = try await autoArima.forecast(horizon: 12)
print("Forecasted points: \(forecast.forecast.predictions)")
```

## Topics

### Automated Time Series
- ``AutoARIMA``
- ``AutoARIMAOrder``
- ``AutoARIMACriterion``
