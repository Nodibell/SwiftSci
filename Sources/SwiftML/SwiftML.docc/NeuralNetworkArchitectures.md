# Multi-Layer Perceptrons & Deep Learning on Apple GPU

Train deep neural network classifiers and regressors using MLX Metal GPU autodiff.

## Overview

`SwiftML` provides `MLPClassifier` and `MLPRegressor`, production-ready multi-layer perceptrons running with native Metal shaders and automatic differentiation on Apple Silicon unified memory.

## 1. Architecture & Activation Functions

$$\mathbf{h}^{(l)} = \sigma\left(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right)$$

Supported activation functions ($\\sigma$):
* **ReLU**: $\max(0, x)$
* **Sigmoid**: $\frac{1}{1 + e^{-x}}$
* **Tanh**: $\frac{e^x - e^{-x}}{e^x + e^{-x}}$
* **Linear / Identity**: $f(x) = x$

## 2. Training an MLP Classifier

```swift
import SwiftML

let mlp = try MLPClassifier(
    hiddenLayerSizes: [64, 32],
    activation: .relu,
    learningRate: 0.001,
    maxEpochs: 100,
    batchSize: 32
)

try await mlp.fit(features: X_train, targets: y_train)

let predictions = try await mlp.predict(features: X_test)
let probabilities = try await mlp.predictProba(features: X_test)
```

## 3. Core ML Binary Export

All MLPs conform to `CoreMLExportable` for direct on-device NeuralNetwork export:

```swift
try await mlp.writeCoreML(
    to: URL(fileURLWithPath: "NeuralNetwork.mlmodel"),
    featureNames: ["feature_1", "feature_2"],
    outputName: "prediction"
)
```

## Topics

### Neural Network Types
- ``MLPClassifier``
- ``MLPRegressor``
