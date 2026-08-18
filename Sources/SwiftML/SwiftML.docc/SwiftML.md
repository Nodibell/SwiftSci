# ``SwiftML``

Supervised Machine Learning Estimators & Neural Networks.

## Overview

`SwiftML` features production-ready supervised learning algorithms, analytical OLS solvers, pre-sorted decision trees, and multi-layer perceptrons.

### Key Capabilities

- **Linear Models**: `LinearRegression` (exact LAPACK OLS solution) and `LogisticRegression` with Metal GPU acceleration.
- **Support Vector Machines**: `LinearSVC` (L2-regularized Hinge loss SVM) with dual CPU and Apple Silicon Metal GPU backends via MLX.
- **Multi-Class Support**: `OneVsRestClassifier` and `LinearSVCOneVsRest` multi-class reduction wrappers.
- **Tree Ensembles**: Pre-sorted `DecisionTreeClassifier`, `RandomForestClassifier`, and `GradientBoostedTreesRegressor`.
- **Deep Learning**: `MLPClassifier` and `MLPRegressor` with BLAS `cblas_dgemm` matrix multiplication and Adam optimizer.
- **Model Export**: ``CoreMLExporter`` producing loadable binary `.mlmodel` files (via ``CoreMLExportable`` protocol) and ``ONNXExporter`` producing standard binary `.onnx` files.

### Example Usage

```swift
import SwiftML

let rf = try RandomForestClassifier(nEstimators: 100, maxDepth: 8)
try await rf.fit(features: X_train, targets: y_train)
let predictions = try await rf.predict(features: X_test)

// Export to binary Core ML
let exportURL = URL(fileURLWithPath: "RandomForest.mlmodel")
try await rf.writeCoreML(to: exportURL, featureNames: ["f1", "f2"], outputName: "label")
```

## Topics

### Guides & Tutorials
- <doc:SupervisedClassifiers>
- <doc:SupervisedRegressors>
- <doc:MLOpsExport>
