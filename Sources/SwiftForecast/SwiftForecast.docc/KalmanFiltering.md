# State-Space Modeling & Joseph-Form Kalman Filters

Perform real-time sensor fusion, state estimation, and noise reduction with numerically stable Kalman Filters.

## Overview

The Kalman Filter is an optimal linear-quadratic recursive estimator for dynamical systems observed through noisy measurements. `SwiftForecast` implements both 1D and multidimensional Kalman filters with **Joseph-form covariance stabilization** to prevent numerical divergence.

## 1. Joseph-Form Covariance Update

Standard covariance update $P_{k|k} = (I - K_k H) P_{k|k-1}$ can become non-positive definite due to floating-point round-off errors. SwiftForecast uses the **Joseph stabilized form**:

$$P_{k|k} = (I - K_k H) P_{k|k-1} (I - K_k H)^T + K_k R_k K_k^T$$

This formulation guarantees that the error covariance matrix remains symmetric and positive-semidefinite under all operating conditions.

## 2. Code Example: 1D Kalman Tracking

```swift
import SwiftForecast

var kf = KalmanFilter1D(
    processVariance: 1e-4,
    measurementVariance: 0.1,
    estimationError: 1.0,
    initialValue: 0.0
)

let noisySignals = [1.02, 0.98, 1.05, 0.95, 1.10, 0.99]
var filteredValues: [Double] = []

for z in noisySignals {
    let state = kf.update(measurement: z)
    filteredValues.append(state)
}

print("Filtered trajectory: \(filteredValues)")
```

## Topics

### Kalman Filter APIs
- ``KalmanFilter1D``
- ``MultidimensionalKalmanFilter``
