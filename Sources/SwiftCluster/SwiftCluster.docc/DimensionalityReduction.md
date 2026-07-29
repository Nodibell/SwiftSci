# Principal Component Analysis (PCA)

Reduce feature dimensionality using SVD-based PCA with automatic CPU and UMA GPU routing.

## Overview

Extract principal components and evaluate explained variance ratios.

### 1. PCA Fitting and Transformation

```swift
import SwiftCluster

let pca = PCA(nComponents: 2, device: .auto)
let X_reduced = try await pca.fitTransform(X)

print("Explained Variance Ratio: \(pca.explainedVarianceRatio ?? [])")
```
