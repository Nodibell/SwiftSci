# Concurrent Model Selection & Cross-Validation

Harness all available Apple Silicon CPU and GPU cores using structured Swift Concurrency `TaskGroup` primitives.

## Overview

Cross-validation and multi-class classification workflows are embarrassingly parallel: each fold split and one-vs-rest binary sub-problem operates on independent slices of tabular data.

`SwiftOptimize` and `SwiftML` leverage structured Swift Concurrency (`withThrowingTaskGroup`) to dispatch multiple fold and class estimators concurrently.

## Structured Concurrency Pipeline

```
           Input Dataset [Features, Targets]
                          │
       ┌──────────────────┴──────────────────┐
       ▼                                     ▼
 Fold 1 Split (Task 1)                 Fold 2 Split (Task 2)
   ├─ Fit Candidate                      ├─ Fit Candidate
   └─ Compute Metric                     └─ Compute Metric
       │                                     │
       └──────────────────┬──────────────────┘
                          ▼
            withThrowingTaskGroup Aggregator
                          ▼
         Cross-Validation Average Leaderboard
```

### AutoML Budgets & Cancellation

`AutoML` deliberately evaluates candidates and folds **sequentially** inside its actor, which keeps ordering deterministic and avoids oversubscribing CPU cores with nested parallel fits. Its `timeBudgetSeconds` is a soft deadline checked before each candidate and fold; a fit already in progress is not interrupted. Parent task cancellation is honoured at the same checkpoints and throws `CancellationError`.

### Deterministic Class Ordering

In `OneVsRestClassifier`, binary classification estimators are dispatched concurrently to maximize throughput across cores. Results are tagged with their class index `(c, est)` and sorted prior to storage, ensuring 100% deterministic prediction ordering across runs.

## Example Usage

```swift
import SwiftOptimize
import SwiftML

// 1. AutoML model selection across Logistic Regression, Decision Tree, Random Forest, and MLP
let automl = AutoML(timeBudgetSeconds: 30.0)
let report = try await automl.fit(features: X, targets: y)
print("Winning Model: \(await automl.bestModelName ?? "-")")
print("CV score: \(report.metrics["cv_score"]!)")

// 2. Parallel One-Vs-Rest classification
let ovr = OneVsRestClassifier(numClasses: 5)
try await ovr.fit(features: X, targets: y)
let predictions = try await ovr.predict(features: X_test)
```

## Topics

### Optimization & Search
- ``AutoML``
- ``AutoMLStrategy``
- ``AutoMLTaskType``
- ``AutoMLLeaderboardEntry``
- ``AutoMLError``
- ``CrossValidationResult``
