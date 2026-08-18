# Exporting Models to Core ML and ONNX

Export trained SwiftSci estimators to native Apple Core ML (`.mlmodel`) and cross-platform ONNX (`.onnx`) binary artifacts for production deployment.

## Overview

SwiftSci provides native, wire-format serialization engines for both Apple Core ML and ONNX without requiring external Python tooling or code generation steps. Models trained natively in Swift can be directly exported to binary artifacts compatible with Apple Neural Engine (ANE), Metal GPU, CPU, and ONNX Runtime.

## 1. Core ML Binary Export (`.mlmodel`)

SwiftSci implements binary Protocol Buffer serialization directly conforming to Apple's `Model.proto` specification (v4).

### Supported Estimators

| Estimator | Core ML Message | Target Hardware |
| :--- | :--- | :--- |
| `LinearRegression` | `GLMRegressor` | CPU / GPU / ANE |
| `LogisticRegression` | `GLMClassifier` | CPU / GPU / ANE |
| `DecisionTreeClassifier` | `TreeEnsembleClassifier` | CPU / GPU / ANE |
| `DecisionTreeRegressor` | `TreeEnsembleRegressor` | CPU / GPU / ANE |
| `RandomForestClassifier` | `TreeEnsembleClassifier` (multi-tree) | CPU / GPU / ANE |
| `RandomForestRegressor` | `TreeEnsembleRegressor` (multi-tree) | CPU / GPU / ANE |

### Using the `CoreMLExportable` Protocol

All supported model types conform to the ``CoreMLExportable`` protocol, providing asynchronous export methods:

```swift
import SwiftML

// Train a Random Forest Classifier
let rf = try RandomForestClassifier(nEstimators: 50, maxDepth: 8)
try await rf.fit(features: X_train, targets: y_train)

// Export directly to a .mlmodel file
let exportURL = URL(fileURLWithPath: "/path/to/RandomForest.mlmodel")
try await rf.writeCoreML(
    to: exportURL,
    featureNames: ["age", "income", "credit_score"],
    outputName: "approved"
)
```

### Direct Serialization via `CoreMLExporter`

You can also use static methods on ``CoreMLExporter`` to produce binary `Data` payloads:

```swift
// Export fitted linear regression
let modelData = CoreMLExporter.exportBinaryLinearModel(
    name: "HousePricePredictor",
    inputNames: ["sqft", "bedrooms", "bathrooms"],
    outputName: "price",
    weights: [250.0, 15000.0, 10000.0],
    bias: 50000.0
)

try modelData.write(to: URL(fileURLWithPath: "HousePricePredictor.mlmodel"))
```

### Loading in Client Applications

Exported `.mlmodel` files can be compiled and loaded dynamically using Apple's `CoreML` framework:

```swift
import CoreML

let compiledURL = try MLModel.compileModel(at: exportURL)
let mlModel = try MLModel(contentsOf: compiledURL)

let inputDict: [String: Any] = ["age": 35.0, "income": 75000.0, "credit_score": 720.0]
let featureProvider = try MLDictionaryFeatureProvider(dictionary: inputDict)
let output = try mlModel.prediction(from: featureProvider)
```

## 2. ONNX Binary Export (`.onnx`)

SwiftSci provides native binary ONNX wire-format export via ``ONNXExporter``:

```swift
import SwiftML

let onnxData = ONNXExporter.exportBinaryONNX(
    name: "LinearONNX",
    inputs: ["features"],
    output: "prediction",
    weights: [1.5, -2.0, 0.75],
    bias: 0.5
)

try onnxData.write(to: URL(fileURLWithPath: "model.onnx"))
```

## Topics

### Core ML Exporter API
- ``CoreMLExportable``
- ``CoreMLExporter``
- ``ONNXExporter``
