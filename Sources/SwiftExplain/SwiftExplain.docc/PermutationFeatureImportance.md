# Permutation Feature Importance & Model Diagnostics

Measure true feature importance by evaluating loss degradation after shuffling feature columns.

## Overview

**Permutation Feature Importance** measures how much a trained model's error increases when a single feature column is randomly shuffled, breaking the relationship between that feature and the target outcome.

$$\text{Importance}(f_j) = L(X_{\text{shuffled}(j)}, y) - L(X, y)$$

## 1. Computing Permutation Importance

```swift
import SwiftExplain
import SwiftML

let importance = try await PermutationImportance.compute(
    model: trainedModel,
    features: X_val,
    targets: y_val,
    metric: .meanSquaredError,
    nRepeats: 10
)

for (idx, score) in importance.enumerated() {
    print("Feature \(idx): Mean drop = \(score.mean), Std = \(score.std)")
}
```

## Topics

### Diagnostic Tools
- ``PermutationImportance``
