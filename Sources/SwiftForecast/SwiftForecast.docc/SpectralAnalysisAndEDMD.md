# Koopman Operator Theory & Spectral Decomposition

Model non-linear dynamical systems using linear infinite-dimensional Koopman operator embeddings.

## Overview

Non-linear time-series systems are notoriously difficult to forecast. **Koopman Operator Theory** maps non-linear state spaces into an infinite-dimensional Hilbert space where the dynamics evolve linearly:

$$g(x_{k+1}) = \mathcal{K} \, g(x_k)$$

`SwiftForecast` implements **Extended Dynamic Mode Decomposition (EDMD)** to approximate the Koopman operator matrix $\mathbf{K}$ using a set of non-linear dictionary observable functions (e.g., polynomials, RBF kernels).

## 1. Fitting an EDMD Model

```swift
import SwiftForecast

// Initialize Koopman Operator with polynomial dictionary observables
let edmd = ExtendedDynamicModeDecomposition(degree: 3)

let trajectory = [1.0, 1.414, 2.0, 2.828, 4.0, 5.656, 8.0]
try await edmd.fit(trajectory: trajectory)

// Predict future states linearly in observable space
let futureSteps = try await edmd.predict(steps: 5)
print("Forecasted states: \(futureSteps)")
```

## Topics

### Spectral Forecasting
- ``ExtendedDynamicModeDecomposition``
