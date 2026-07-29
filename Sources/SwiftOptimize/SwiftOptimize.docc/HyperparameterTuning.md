# Hyperparameter Tuning & AutoML

Automate parameter optimization with GridSearchCV, RandomizedSearchCV, and AutoML model selection.

## Overview

Systematically evaluate parameter grids to discover peak validation performance.

### 1. Grid Search CV

```swift
import SwiftOptimize
import SwiftML

let paramGrid: [String: [Any]] = [
    "maxDepth": [3, 5, 10],
    "minSamplesSplit": [2, 5]
]

let gridSearch = GridSearchCV(estimator: DecisionTreeClassifier(), paramGrid: paramGrid, cv: 5)
try await gridSearch.fit(features: X, targets: y)

print("Best Parameters: \(gridSearch.bestParams)")
```
