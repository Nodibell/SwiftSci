# Feature Scaling & Normalization

Standardize and scale feature matrices using StandardScaler, MinMaxScaler, RobustScaler, and Normalizer.

## Overview

Feature scaling ensures zero-mean, unit-variance, or bounded feature ranges for gradient-based and distance-based estimators.

### 1. StandardScaler

```swift
import SwiftPreprocessing

let X: [[Double]] = [
    [1.0, 200.0],
    [2.0, 300.0],
    [3.0, 400.0]
]

let scaler = StandardScaler()
let X_scaled = try scaler.fitTransform(X)
```

### 2. MinMaxScaler

```swift
let minMax = MinMaxScaler(featureRange: (0.0, 1.0))
let X_norm = try minMax.fitTransform(X)
```
