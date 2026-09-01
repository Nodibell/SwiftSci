# Time Series Forecasting, Smoothing & Kalman Filtering

Model trends, seasonality, volatility, and dynamic state transitions using `ExponentialSmoothing`, `NelderMead`, `KalmanFilter`, `GARCHModel`, and `TimeSeriesDecomposition`.

## Overview

Temporal data analysis requires separating underlying long-term trends, periodic seasonal patterns, and random innovations. `SwiftForecast` provides hardware-accelerated time-series models leveraging Apple Accelerate FFT convolution and LAPACK linear solvers.

---

## 1. Holt-Winters Exponential Smoothing & Nelder-Mead Optimization

`ExponentialSmoothing` models level ($\alpha$), trend ($\beta$), and seasonal ($\gamma$) components with additive or multiplicative adjustments:

$$\hat{y}_{t+h|t} = \ell_t + h b_t + s_{t+h-m(k+1)}$$

Parameters $\alpha, \beta, \gamma$ are automatically optimized using the **Nelder-Mead simplex optimizer** minimizing Mean Squared Error (MSE):

```swift
import Foundation
import SwiftForecast

// 1. Synthetic monthly seasonal dataset (36 observations, 3 years)
let series = (0..<36).map { t in
    let trend = 10.0 + Double(t) * 0.5
    let seasonal = sin(Double(t % 12) * 2.0 * Double.pi / 12.0) * 3.0
    return trend + seasonal + Double.random(in: -0.2...0.2)
}

// 2. Configure Triple Exponential Smoothing (Holt-Winters)
let model = ExponentialSmoothing(
    trend: .additive,
    seasonal: .additive,
    seasonalPeriods: 12
)

// 3. Fit parameters via continuous Nelder-Mead Simplex search
try model.fit(series: series)

if let alpha = model.alpha, let beta = model.beta, let gamma = model.gamma {
    print("Optimized Parameters: alpha=\(String(format: "%.3f", alpha)), beta=\(String(format: "%.3f", beta)), gamma=\(String(format: "%.3f", gamma))")
}

// 4. Forecast future periods
let forecast = try model.forecast(steps: 12)
print("12-Month Out-of-Sample Forecast: \(forecast)")
```

---

## 2. High-Performance 1D Kalman Filter (LAPACK `dgesv`)

`KalmanFilter` estimates the hidden state vector of a linear dynamic system over time from noisy observations. The implementation uses contiguous 1D flat buffers with Apple Accelerate LAPACK `dgesv` matrix inversions:

$$\mathbf{x}_{k|k-1} = \mathbf{F}_k \mathbf{x}_{k-1|k-1} + \mathbf{B}_k \mathbf{u}_k$$
$$\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^T + \mathbf{Q}_k$$
$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^T (\mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k)^{-1}$$

```swift
import SwiftForecast

// 1. Configure 2D state system: [position, velocity] with 1D observation [position]
let stateDim = 2
let obsDim = 1

// State Transition Matrix F: position += dt * velocity
let F: [Double] = [
    1.0, 1.0,  // row 0: pos = 1.0*pos + 1.0*vel
    0.0, 1.0   // row 1: vel = 0.0*pos + 1.0*vel
]

// Observation Matrix H: we measure position
let H: [Double] = [1.0, 0.0]

// Process Noise Q and Measurement Noise R
let Q: [Double] = [
    0.01, 0.0,
    0.0,  0.01
]
let R: [Double] = [0.1]

// Initial State and Covariance
let x0: [Double] = [0.0, 1.0]
let P0: [Double] = [
    1.0, 0.0,
    0.0, 1.0
]

var kf = try KalmanFilter(
    stateDim: stateDim,
    obsDim: obsDim,
    transitionMatrix: F,
    observationMatrix: H,
    processNoise: Q,
    measurementNoise: R,
    initialState: x0,
    initialCovariance: P0
)

// 2. Track real-time noisy trajectory
let noisyMeasurements: [[Double]] = [[0.95], [2.05], [2.98], [4.12], [5.01]]

print("=== Kalman Filtering Updates ===")
for (step, z) in noisyMeasurements.enumerated() {
    try kf.predict()
    try kf.update(observation: z)
    
    let state = kf.state
    print("Step \(step + 1): Position=\(String(format: "%.3f", state[0])), Velocity=\(String(format: "%.3f", state[1]))")
}
```

---

## 3. Classical Seasonal Decomposition (STL)

Separate series into Trend, Seasonal, and Residual components:

```swift
let decomp = TimeSeriesDecomposition.decompose(series: series, period: 12, type: .additive)

print("Extracted Trend Component count: \(decomp.trend.compactMap { $0 }.count)")
print("Seasonal Component Cycle: \(decomp.seasonal.prefix(12))")
```
