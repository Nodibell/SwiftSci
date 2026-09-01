# Categorical Encoders & Missing Value Imputation

Encode non-numeric categories into numerical representations and impute missing values.

## Overview

`SwiftPreprocessing` provides industry-standard transformers with full Scikit-Learn parity:

## 1. One-Hot Encoding

Transforms categorical variables into binary indicator vectors:

```swift
import SwiftPreprocessing
import SwiftDataFrame

let categories = ["Red", "Green", "Blue", "Red"]
let encoder = OneHotEncoder()
try await encoder.fit(categories)

let encodedMatrix = try await encoder.transform(["Blue", "Red"])
// Returns [[0, 0, 1], [1, 0, 0]]
```

## 2. Standard Scaling & Robust Scaling

> **Standard Scaling:** `z = (x - μ) / σ`  
> **Robust Scaling:** `z_robust = (x - Median) / IQR`

```swift
let scaler = StandardScaler()
try await scaler.fit(features: X_train)
let X_scaled = try await scaler.transform(features: X_train)
```

## Topics

### Preprocessing Transformers
- ``StandardScaler``
- ``MinMaxScaler``
- ``RobustScaler``
- ``OneHotEncoder``
