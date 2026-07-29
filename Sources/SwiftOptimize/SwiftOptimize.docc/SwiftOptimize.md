# ``SwiftOptimize``

AutoML, Cross-Validation Folds & Hyperparameter Tuning.

## Overview

`SwiftOptimize` enables automated model selection, advanced cross-validation splitters, and systematic hyperparameter search strategies.

### Key Capabilities

- **Cross-Validation Splitters**: `KFold`, `StratifiedKFold`, `GroupKFold`, and `TimeSeriesSplit`.
- **Hyperparameter Search**: `GridSearchCV` and `RandomizedSearchCV` with parallel fold evaluation.
- **Automated Machine Learning**: `AutoML` model selector with automatic cross-validation tuning.
- **Evaluation Metrics**: `fBetaScore`, `prAUC`, `adjustedR2Score`, `mape`, and `explainedVarianceScore`.

### Example Usage

```swift
import SwiftOptimize

let skf = StratifiedKFold(nSplits: 5)
let gridSearch = GridSearchCV(estimator: DecisionTreeClassifier(), paramGrid: grid, cv: 5)
```

## Topics

### Guides & Tutorials
- <doc:CrossValidationFolds>
- <doc:HyperparameterTuning>
