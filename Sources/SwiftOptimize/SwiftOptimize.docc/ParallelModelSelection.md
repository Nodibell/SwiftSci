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

### Cooperative Cancellation & Bounded Budgets

When training under a time budget (such as `AutoML(timeBudgetSeconds:)`), tasks run inside a managed scope. If the time budget expires or a fatal error occurs in a critical pipeline step, child tasks cooperatively respect task cancellation without leaking threads or memory.

### Deterministic Class Ordering

In `OneVsRestClassifier`, binary classification estimators are dispatched concurrently to maximize throughput across cores. Results are tagged with their class index `(c, est)` and sorted prior to storage, ensuring 100% deterministic prediction ordering across runs.

## Example Usage

```swift
import SwiftOptimize
import SwiftML

// 1. Concurrent AutoML search across DecisionTree, RandomForest, and MLP
let automl = AutoML(timeBudgetSeconds: 30.0)
let report = try await automl.fit(features: X, targets: y)
print("Winning Model: \(report.bestModel)")

// 2. Parallel One-Vs-Rest classification
let ovr = OneVsRestClassifier(numClasses: 5)
try await ovr.fit(features: X, targets: y)
let predictions = try await ovr.predict(features: X_test)
```

## Topics

### Optimization & Search
- ``AutoML``
- ``AutoMLStrategy``
- ``CrossValidationResult``
