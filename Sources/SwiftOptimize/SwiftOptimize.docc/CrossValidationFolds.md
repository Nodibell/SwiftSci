# Cross-Validation Strategies

Evaluate model stability across dataset splits using KFold, StratifiedKFold, GroupKFold, and TimeSeriesSplit.

## Overview

Split datasets safely without data leakage across classes, groups, or temporal sequences.

### 1. Stratified K-Fold

```swift
import SwiftOptimize

let skf = StratifiedKFold(nSplits: 5, shuffle: true, randomSeed: 42)
for (fold, (trainIdx, valIdx)) in skf.split(X: X, y: y).enumerated() {
    print("Fold \(fold): Train size=\(trainIdx.count), Val size=\(valIdx.count)")
}
```
