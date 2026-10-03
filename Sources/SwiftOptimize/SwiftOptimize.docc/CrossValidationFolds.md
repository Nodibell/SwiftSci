# Cross-Validation Strategies

Evaluate model stability and generalization performance across dataset splits using `KFold`, `StratifiedKFold`, and `TimeSeriesSplit`.

## Overview

Cross-validation is essential for estimating the out-of-sample prediction error and hyperparameter tuning without data leakage. `SwiftOptimize` generates zero-allocation index pairs `(trainIndices, validationIndices)` instead of duplicating dataset matrices in memory.

### Index-Based Split Architecture

Each split strategy implements deterministic or seeded index generation:

```
Dataset (N samples)
┌─────────────────────────────────────────────────────────────┐
│ Fold 1: [ Val (20%) ] [                Train (80%)        ] │
│ Fold 2: [ Train (20%) ] [ Val (20%) ] [ Train (60%)       ] │
│ Fold 3: [ Train (40%) ] [ Val (20%) ] [ Train (40%)       ] │
│ Fold 4: [ Train (60%) ] [ Val (20%) ] [ Train (20%)       ] │
│ Fold 5: [                Train (80%)        ] [ Val (20%) ] │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Standard K-Fold Cross-Validation

`KFold` splits continuous regression data or balanced observations into $K$ contiguous or randomized partitions.

```swift
import Foundation
import SwiftDataFrame
import SwiftOptimize
import SwiftML

// 1. Prepare synthetic features and targets
let sampleCount = 100
let featureCount = 4
let X = (0..<sampleCount).map { row in
    (0..<featureCount).map { col in Double(row * featureCount + col) * 0.05 }
}
let y = X.map { row in row.reduce(0.0, +) * 1.5 + 2.0 }

// 2. Configure 5-fold cross-validation with a fixed random seed
let kfold = try KFold(nSplits: 5, shuffle: true, seed: 42)
let splits = try kfold.split(features: X, targets: y)

print("Generated \(splits.count) validation folds.")

// 3. Train and evaluate linear regressors across folds
var mseScores: [Double] = []

for (foldIndex, fold) in splits.enumerated() {
    let trainX = fold.trainFeatures
    let trainY = fold.trainTargets
    let valX = fold.valFeatures
    let valY = fold.valTargets
    
    let model = LinearRegression(learningRate: 0.01, epochs: 200)
    try await model.fit(features: trainX, targets: trainY)
    
    let predictions = try await model.predict(features: valX)
    let mse = Metrics.meanSquaredError(yTrue: valY, yPred: predictions)
    mseScores.append(mse)
    
    print("Fold \(foldIndex + 1): MSE = \(String(format: "%.4f", mse))")
}

let meanMSE = mseScores.reduce(0.0, +) / Double(mseScores.count)
print("Average Cross-Validation MSE: \(String(format: "%.4f", meanMSE))")
```

---

## 2. Stratified K-Fold for Classification

`StratifiedKFold` ensures that every validation fold preserves the exact class label distribution proportions of the entire dataset. This is critical for imbalanced target classes.

```swift
import SwiftOptimize
import SwiftML

// 1. Dataset with imbalanced binary classes (80% Class 0, 20% Class 1)
let features = (0..<100).map { i in [Double(i), Double(i * 2)] }
let labels = (0..<100).map { $0 < 80 ? 0.0 : 1.0 }

// 2. Initialize Stratified K-Fold
let stratifiedKF = try StratifiedKFold(nSplits: 5, shuffle: true, seed: 123)
let folds = try stratifiedKF.split(features: features, targets: labels)

for (idx, fold) in folds.enumerated() {
    let valLabels = fold.valTargets.map { Int($0) }
    let class1Count = valLabels.filter { $0 == 1 }.count
    let class0Count = valLabels.filter { $0 == 0 }.count
    
    print("Fold \(idx + 1): Class 0 = \(class0Count), Class 1 = \(class1Count) (Ratio: \(Double(class1Count)/Double(valLabels.count)))")
}
```

---

## 3. Expanding-Window TimeSeriesSplit

For temporal and sequential datasets, standard random cross-validation introduces data leakage (evaluating on historical data after training on future observations). `TimeSeriesSplit` creates expanding historical training windows with forward-looking validation sets.

```swift
import SwiftOptimize
import SwiftForecast

let timeSeriesData = (0..<120).map { i in Double(i) * 1.2 + sin(Double(i) * 0.2) }

let tsSplit = try TimeSeriesSplit(nSplits: 4, maxTrainSize: nil)
let temporalFolds = try tsSplit.split(features: timeSeriesData.map { [$0] }, targets: timeSeriesData)

for (foldIndex, fold) in temporalFolds.enumerated() {
    print("Fold \(foldIndex + 1): Train \(fold.trainTargets.count) pts -> Val \(fold.valTargets.count) pts")
}
```

> **Performance Tip:**
> Always access data via index arrays (`trainIndices.map { X[$0] }`) rather than cloning full matrices up front. This keeps memory footprints proportional to the active batch size.
