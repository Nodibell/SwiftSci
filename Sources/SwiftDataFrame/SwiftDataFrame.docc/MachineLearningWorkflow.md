# Machine Learning Workflow with SwiftML

End-to-end Machine Learning pipelines using `SwiftML`, `SwiftPreprocessing`, and `SwiftCluster`.

## Overview

`SwiftML` provides vectorized classification, regression, clustering, and neural network algorithms using Apple Accelerate (`vDSP`, `BLAS`, `LAPACK`).

### 1. Preprocessing Data

```swift
import SwiftDataFrame
import SwiftPreprocessing

let scaler = StandardScaler()
let scaledX = try scaler.fitTransform(features)

let split = try trainTestSplit(X: scaledX, y: targets, testSize: 0.2, seed: 42)
```

### 2. Training Multi-Layer Perceptron (MLP)

```swift
import SwiftML

let mlp = MLPClassifier(
    hiddenLayerSizes: [64, 32],
    maxIter: 200,
    learningRate: 0.01,
    seed: 42
)

try await mlp.fit(features: split.trainX, targets: split.trainY)
let predictions = try await mlp.predict(features: split.testX)
```

### 3. Dimensionality Reduction & Clustering

```swift
import SwiftCluster

let pca = PCA(nComponents: 2)
let reducedX = try pca.fitTransform(split.trainX)

let kmeans = KMeans(k: 3)
try kmeans.fit(reducedX)
let labels = kmeans.labels
```
