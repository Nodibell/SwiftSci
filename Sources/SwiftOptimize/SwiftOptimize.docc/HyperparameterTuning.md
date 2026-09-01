# Hyperparameter Tuning & AutoML Engine

Automate model exploration and parameter optimization with `GridSearchCV`, `RandomizedSearchCV`, and the intelligent `AutoML` model selector.

## Overview

Finding the optimal combination of hyperparameters is critical for maximizing generalization accuracy. `SwiftOptimize` provides exhaustive grid search, randomized sampling for high-dimensional parameter spaces, and end-to-end `AutoML` selection across multiple model families.

---

## 1. AutoML — Automated Model & Parameter Selection

`AutoML` evaluates candidate model architectures (Decision Trees, Random Forests, Logistic Regression, Linear Regression, and Multi-Layer Perceptrons) using cross-validation, ranking candidates by performance metrics ($F_1$-score for classification, $R^2$ for regression).

```swift
import Foundation
import SwiftOptimize
import SwiftML

// 1. Synthetic classification dataset
let N = 120
let D = 4
let X = (0..<N).map { i in
    (0..<D).map { d in Double(i * D + d) * 0.05 + sin(Double(i + d)) }
}
let y = (0..<N).map { i in i % 2 }

// 2. Configure AutoML with 3-fold cross validation
let autoML = AutoML(
    task: .classification,
    timeLimitSeconds: 30,
    cvFolds: 3,
    metric: "f1"
)

// 3. Train and select the optimal estimator
let result = try await autoML.fit(features: X, targets: y)

print("=== AutoML Optimization Results ===")
print("Selected Best Model: \(result.bestModelName)")
print("Validation F1 Score: \(String(format: "%.4f", result.bestScore))")
print("Tuned Hyperparameters: \(result.bestParameters)")

// 4. Run inference directly on new test samples
let newSample = [0.5, 1.2, 0.3, 0.8]
let testPrediction = try await result.bestEstimator.predict(features: [newSample])
print("Prediction: \(testPrediction)")
```

---

## 2. Exhaustive Grid Search (`GridSearchCV`)

`GridSearchCV` exhaustively tests all Cartesian combinations of hyperparameter discrete values.

```swift
import SwiftOptimize
import SwiftML

// Define hyperparameter search grid for DecisionTreeClassifier
let paramGrid: [String: [Any]] = [
    "maxDepth": [3, 5, 8],
    "minSamplesSplit": [2, 5, 10],
    "criterion": ["gini", "entropy"]
]

let gridSearch = GridSearchCV(
    estimator: DecisionTreeClassifier(maxDepth: 3),
    paramGrid: paramGrid,
    cv: 5,
    scoring: .accuracy
)

try await gridSearch.fit(features: X, targets: y)

if let bestParams = gridSearch.bestParams {
    print("Optimal Depth: \(bestParams["maxDepth"] ?? "default")")
    print("Peak CV Accuracy: \(String(format: "%.4f", gridSearch.bestScore ?? 0.0))")
}
```

---

## 3. Randomized Search (`RandomizedSearchCV`)

When searching through large continuous or discrete combinatorial spaces, `RandomizedSearchCV` samples $N$ random configurations, finding near-optimal solutions in a fraction of the compute time.

```swift
import SwiftOptimize
import SwiftML

let distributions: [String: [Any]] = [
    "learningRate": [0.001, 0.005, 0.01, 0.05, 0.1],
    "epochs": [100, 200, 500, 1000],
    "l2Penalty": [0.0, 0.0001, 0.001, 0.01]
]

let randomSearch = RandomizedSearchCV(
    estimator: LogisticRegression(learningRate: 0.01),
    paramDistributions: distributions,
    nIter: 20,
    cv: 3,
    randomSeed: 42
)

try await randomSearch.fit(features: X, targets: y)
print("Top Configuration Score: \(randomSearch.bestScore ?? 0.0)")
```
