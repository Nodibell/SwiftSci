# Supervised Regression Models

Perform continuous target predictions using LinearRegression (exact OLS), DecisionTreeRegressor, and GradientBoostedTreesRegressor.

## Overview

Optimized regression solvers leveraging LAPACK linear algebra and decision trees.

### 1. Linear Regression (Exact OLS)

```swift
import SwiftML

let reg = LinearRegression(device: .auto)
try await reg.fit(features: X_train, targets: y_train)

let (weights, bias) = reg.getWeightsAndBias()
print("Weights: \(weights ?? []), Bias: \(bias ?? 0.0)")
```

### 2. Gradient Boosted Trees Regressor

```swift
let gbdt = GradientBoostedTreesRegressor(nEstimators: 50, learningRate: 0.1, maxDepth: 4)
try await gbdt.fit(features: X_train, targets: y_train)
let preds = try await gbdt.predict(features: X_test)
```
