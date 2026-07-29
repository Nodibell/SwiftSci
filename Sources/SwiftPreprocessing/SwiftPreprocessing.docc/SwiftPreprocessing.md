# ``SwiftPreprocessing``

Feature Scaling, Categorical Encoders & Preprocessing Pipelines.

## Overview

`SwiftPreprocessing` offers scikit-learn parity transformers for feature scaling, encoding, missing value imputation, and pipeline orchestration.

### Key Capabilities

- **Feature Scalers**: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, and `Normalizer`.
- **Categorical Encoders**: `OneHotEncoder`, `TargetEncoder`, `OrdinalEncoder`, and `FrequencyEncoder`.
- **Missing Value Imputation**: `Imputer` (mean/median/mode) and multi-variate `KNNImputer`.
- **Pipeline Orchestration**: `Pipeline` and `ColumnTransformer` for multi-type feature engineering workflows.

### Example Usage

```swift
import SwiftPreprocessing

let scaler = StandardScaler()
let X_scaled = try scaler.fitTransform(X)
```

## Topics

### Guides & Tutorials
- <doc:FeatureScaling>
- <doc:EncodersAndImputers>
