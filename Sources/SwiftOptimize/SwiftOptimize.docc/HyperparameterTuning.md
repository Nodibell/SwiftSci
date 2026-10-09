
# Hyperparameter Tuning & AutoML Engine

Automate model exploration and parameter optimization with `GridSearchCV`, `RandomizedSearchCV`, and the intelligent `AutoML` model selector.

## Overview

Finding the optimal combination of hyperparameters is critical for maximizing generalization accuracy. `SwiftOptimize` provides exhaustive grid search, randomized sampling for high-dimensional parameter spaces, and end-to-end `AutoML` selection across multiple model families.

---

## 1. AutoML — Automated Model Selection

`AutoML` cross-validates a fixed set of baseline architectures (Logistic/Linear Regression, Decision Tree, Random Forest, MLP) and ranks them by the **mean per-fold** score: macro F1 for classification, R² for regression. Classification uses ``StratifiedKFold``; regression uses shuffled ``KFold``. All metrics come from ``EvaluationHarness``.

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
let y = (0..<N).map { i in Double(i % 2) }

// 2. Configure AutoML with 3-fold stratified cross-validation
let autoML = AutoML(
    timeBudgetSeconds: 30,
    taskType: .classification,
    nFolds: 3,
    seed: 42
)

// 3. Run model selection
let report = try await autoML.fit(features: X, targets: y)

print("Best model: \(await autoML.bestModelName ?? "-")")
print("CV macro F1: \(report.metrics["cv_score"]!) ± \(report.metrics["cv_std"]!)")

for entry in await autoML.leaderboard {
    print(entry.displayName, entry.meanScore, entry.foldScores)
}
```

- **Task type:** `.auto` (default) classifies integer targets with ≤ 10 distinct values, treats non-integer or high-cardinality targets as regression, and throws ``AutoMLError/ambiguousTaskType(uniqueValues:sampleCount:)`` in between. Pass `taskType` explicitly in that case.
- **Time budget:** soft. It is checked before each candidate and fold; a running fit is not interrupted, and the first candidate always completes.
- **Search:** only `.modelSelection` is available. To tune hyperparameters of a single estimator, use ``GridSearchCV`` or ``RandomizedSearchCV`` below.

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
