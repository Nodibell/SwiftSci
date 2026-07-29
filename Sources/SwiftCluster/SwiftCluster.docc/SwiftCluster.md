# ``SwiftCluster``

PCA Dimensionality Reduction & Unsupervised Clustering.

## Overview

`SwiftCluster` provides fast SVD-based dimensionality reduction, density-based clustering, and anomaly detection algorithms.

### Key Capabilities

- **Dimensionality Reduction**: SVD-based `PCA` with hardware auto-routing between CPU and UMA GPU.
- **Clustering**: Accelerated `KMeans` with KMeans++ initialization and `DBSCAN` density-based clustering.
- **Anomaly Detection**: `IsolationForest` and `LocalOutlierFactor` (LOF) density outlier identification.
- **Clustering Metrics**: `SilhouetteScore`, Calinski-Harabasz Index, and Davies-Bouldin Index.

### Example Usage

```swift
import SwiftCluster

let pca = PCA(nComponents: 2)
let X_pca = try await pca.fitTransform(X)
```

## Topics

### Guides & Tutorials
- <doc:DimensionalityReduction>
- <doc:ClusteringAndOutliers>
