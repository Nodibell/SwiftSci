# SHAP Feature Attributions

Compute exact and approximate Shapley feature importance scores using parallel KernelSHAP and tree-accelerated TreeSHAP.

## Overview

Explain model predictions with game-theoretic Shapley value attributions.

### 1. TreeSHAP for Tree Models

```swift
import SwiftExplain
import SwiftML

let model = DecisionTreeClassifier(maxDepth: 5)
try await model.fit(features: X_train, targets: y_train)

let treeShap = TreeSHAP()
let shapValues = try treeShap.explain(model: model, instance: X_test[0])
print("Feature SHAP values: \(shapValues)")
```
