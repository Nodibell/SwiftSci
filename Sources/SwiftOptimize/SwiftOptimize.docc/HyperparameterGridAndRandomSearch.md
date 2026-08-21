# Hyperparameter Tuning with Grid & Random Search

Optimize machine learning model architectures using cross-validated search strategies.

## Overview

Finding optimal hyperparameters (e.g. tree depth, learning rates, regularization constants) is crucial for maximizing predictive performance. `SwiftOptimize` provides parallelized **Grid Search** and **Random Search** cross-validation engines.

## 1. Grid Search Cross-Validation

```swift
import SwiftOptimize
import SwiftML

let paramGrid: [String: [Any]] = [
    "nEstimators": [50, 100, 200],
    "maxDepth": [4, 8, 12]
]

let search = GridSearchCV(
    estimator: RandomForestClassifier(),
    paramGrid: paramGrid,
    cv: KFold(nSplits: 5, shuffle: true, randomSeed: 42),
    scoring: .rocAuc
)

try await search.fit(features: X_train, targets: y_train)

print("Best Parameters: \(search.bestParams)")
print("Best Score (ROC-AUC): \(search.bestScore)")
```

## Topics

### Optimization Types
- ``GridSearchCV``
- ``RandomizedSearchCV``
- ``KFold``
