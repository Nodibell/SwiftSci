# ``SwiftExplain``

Model Explainability, SHAP & Feature Attributions.

## Overview

`SwiftExplain` provides game-theoretic feature attributions and diagnostic plots to interpret complex black-box models.

### Key Capabilities

- **SHAP Value Estimation**: Parallel `KernelSHAP` for arbitrary estimators.
- **Tree-Accelerated Attributions**: Fast `TreeSHAP` for decision trees and random forests.
- **Feature Importance**: Model-agnostic `PermutationImportance` evaluation.
- **Partial Dependence**: `PartialDependencePlot` (PDP) for marginal feature response curves.

### Example Usage

```swift
import SwiftExplain

let treeShap = TreeSHAP()
let attributions = try treeShap.explain(model: model, instance: X[0])
```

## Topics

### Guides & Tutorials
- <doc:ShapleyAttributions>
- <doc:ModelInterpretability>
