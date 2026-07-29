# Categorical Encoding & Imputation

Encode categorical features and impute missing data with OneHotEncoder, TargetEncoder, KNNImputer, and ColumnTransformer.

## Overview

Transform non-numeric columns into numerical representations and handle missing data seamlessly.

### 1. OneHotEncoder

```swift
import SwiftPreprocessing

let categories = ["Red", "Green", "Blue", "Red"]
let encoder = OneHotEncoder()
let encoded = try encoder.fitTransform(categories)
```

### 2. ColumnTransformer Pipeline

```swift
let transformer = ColumnTransformer(transformers: [
    ("num", StandardScaler(), ["Age", "Fare"]),
    ("cat", OneHotEncoder(), ["Sex", "Embarked"])
])
```
