# ``SwiftML``

Supervised Machine Learning Estimators & Neural Networks.

## Overview

`SwiftML` features production-ready supervised learning algorithms, analytical OLS solvers, pre-sorted decision trees, and multi-layer perceptrons.

### Key Capabilities

- **Linear Models**: `LinearRegression` (exact LAPACK OLS solution) and `LogisticRegression`.
- **Tree Ensembles**: Pre-sorted `DecisionTreeClassifier`, `RandomForestClassifier`, and `GradientBoostedTreesRegressor`.
- **Multi-Class Support**: `OneVsRestClassifier` multi-class reduction solver.
- **Deep Learning**: `MLPClassifier` and `MLPRegressor` with BLAS `cblas_dgemm` matrix multiplication and Adam optimizer.

### Example Usage

```swift
import SwiftML

let rf = RandomForestClassifier(nEstimators: 100, maxDepth: 8)
try await rf.fit(features: X_train, targets: y_train)
let predictions = try await rf.predict(features: X_test)
```

## Topics

### Guides & Tutorials
- <doc:SupervisedClassifiers>
- <doc:SupervisedRegressors>
