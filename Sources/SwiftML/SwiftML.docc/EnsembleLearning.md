# Ensemble Learning: Random Forests & Gradient Boosted Trees

Train high-accuracy bagging and boosting decision tree ensembles natively in Swift with multicore parallelism.

## Overview

Ensemble methods combine multiple base estimators to improve generalization and reduce variance. `SwiftML` implements both **Random Forests** (bagging) and **Gradient Boosted Decision Trees** (boosting) with high-performance tree split search.

## 1. Random Forest Classification & Regression

`RandomForestClassifier` trains an ensemble of randomized decision trees using bootstrap aggregation (bagging) and random feature subsets:

```swift
import SwiftML

let rf = try RandomForestClassifier(
    nEstimators: 100,
    maxDepth: 10,
    minSamplesSplit: 2,
    criterion: .gini
)

try await rf.fit(features: X_train, targets: y_train)
let predictions = try await rf.predict(features: X_test)
let probabilities = try await rf.predictProba(features: X_test)
```

## 2. Gradient Boosted Decision Trees (GBDT)

`GradientBoostingClassifier` sequentially trains decision trees to minimize pseudo-residuals using gradient descent in function space:

```swift
let gbdt = try GradientBoostingClassifier(
    nEstimators: 50,
    learningRate: 0.1,
    maxDepth: 5
)

try await gbdt.fit(features: X_train, targets: y_train)
```

## 3. Feature Importance Extraction

Both tree models compute Gini/MSE feature importances directly:

```swift
let importances = rf.featureImportances
print("Feature importances: \(importances)")
```

## Topics

### Tree Estimators
- ``RandomForestClassifier``
- ``RandomForestRegressor``
- ``GradientBoostingClassifier``
- ``GradientBoostingRegressor``
