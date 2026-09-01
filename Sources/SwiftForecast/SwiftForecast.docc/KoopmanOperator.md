# ``SwiftForecast/KoopmanOperator``

Nonlinear Dynamic System Forecasting via Extended Dynamic Mode Decomposition (EDMD).

## Overview

The `KoopmanOperator` presents an infinite-dimensional linear operator framework for forecasting nonlinear time series and complex dynamical systems without explicit physical differential equations.

### Features

- **Observable Dictionaries**: Polynomial, Radial Basis Functions (RBF), Fourier trigonometric, and custom/combined dictionaries (`ObservableDictionary`).
- **Hankel Time-Delay Embedding**: Reconstructs hidden attractor manifolds for 1D scalar time series.
- **Extended Dynamic Mode Decomposition (EDMD)**: Ridge-regularized linear operator regression in lifted observable spaces.
- **Spectral Stability Analysis**: Computes complex eigenvalues `λ_i` via LAPACK `dgeev` for stability and frequency decomposition.

### Example Usage

```swift
import SwiftForecast

// Fit Koopman Operator on a 1D non-linear time series
let koopman = KoopmanOperator(
    dictionary: .polynomial(degree: 2),
    regularization: 1e-5,
    embeddingLags: 3
)

try await koopman.fit(series: timeSeriesData)

// Predict 10 steps ahead
let predictions = try await koopman.predict1D(horizon: 10)

// Analyze system stability via complex eigenvalues
let eigenvalues = try await koopman.eigenvalues()
```
