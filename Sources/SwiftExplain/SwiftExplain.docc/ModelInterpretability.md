# Partial Dependence & Permutation Importance

Evaluate feature relationships and global importance via Partial Dependence Plots (PDP) and Permutation Importance.

## Overview

Quantify the global contribution of features and plot marginal responses.

### 1. Permutation Feature Importance

```swift
import SwiftExplain

let permImportance = PermutationImportance()
let importances = try await permImportance.compute(model: model, features: X_test, targets: y_test)
print("Feature Importances: \(importances)")
```
