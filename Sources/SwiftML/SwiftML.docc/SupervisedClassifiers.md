# Supervised Classification Models

Train classification models using LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, and MLPClassifier.

## Overview

`SwiftML` provides production-ready classification estimators with parallel training and evaluation.

### 1. Random Forest Classifier

```swift
import SwiftML

let rf = RandomForestClassifier(nEstimators: 100, maxDepth: 10)
try await rf.fit(features: X_train, targets: y_train)

let predictions = try await rf.predict(features: X_test)
let probabilities = try await rf.predictProbability(features: X_test)
```

### 2. LinearSVC / Support Vector Classifier

`LinearSVC` uses L2-regularized Hinge loss with automatic Apple Silicon Metal GPU acceleration:

```swift
import SwiftML

// Binary SVC
let svc = LinearSVC(C: 1.0, device: .auto)
try await svc.fit(features: X_train, targets: y_train)
let predictions = try await svc.predict(features: X_test)

// Multi-class (One-Vs-Rest)
let ovrSVC = LinearSVCOneVsRest(numClasses: 5)
try await ovrSVC.fit(features: X_train, targets: y_train)
let multiPredictions = try await ovrSVC.predict(features: X_test)
```

### 3. Multi-Layer Perceptron (MLP)

```swift
let mlp = MLPClassifier(hiddenLayerSizes: [64, 32], activation: .relu, maxEpochs: 200)
try await mlp.fit(features: X_train, targets: y_train)
```

### 4. Binary Core ML Export

Fitted classifiers conforming to ``CoreMLExportable`` can be exported directly to `.mlmodel`:

```swift
let exportURL = URL(fileURLWithPath: "Classifier.mlmodel")
try await rf.writeCoreML(
    to: exportURL,
    featureNames: ["feature_1", "feature_2"],
    outputName: "label"
)
```

For full details, see <doc:MLOpsExport>.
