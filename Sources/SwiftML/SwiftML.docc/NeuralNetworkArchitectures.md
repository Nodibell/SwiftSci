# Multi-Layer Perceptrons & Neural Network Architecture

Train neural network classifiers and regressors with hardware-accelerated BLAS forward passes, Adam optimization, Dropout regularization, and Batch Normalization.

## Overview

`SwiftML` provides `MLPClassifier` and `MLPRegressor` multi-layer perceptrons designed for high-throughput tabular modeling. The architecture uses contiguous `LayerWeights` flat buffers with Apple Accelerate BLAS `cblas_dgemm` matrix multiplications and vectorized Adam optimizer updates.

```
Input Layer (D features)
      │
      ▼
┌──────────────┐
│ Linear (W1)  │ ───► Apple Accelerate cblas_dgemm
├──────────────┤
│  BatchNorm   │ ───► (x - mean) / sqrt(var + eps)
├──────────────┤
│  Activation  │ ───► ReLU / Tanh / Sigmoid
├──────────────┤
│   Dropout    │ ───► Bernoulli Mask (p = 0.2)
└──────────────┘
      │
      ▼
┌──────────────┐
│ Linear (W2)  │ ───► cblas_dgemm
├──────────────┤
│  Softmax/Lin │ ───► Probabilities / Continuous Output
└──────────────┘
```

---

## 1. Mathematical Formulation

For each hidden layer $l$:
$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$
$$\mathbf{a}^{(l)} = \sigma\left(\mathbf{z}^{(l)}\right)$$

Supported activations ($\sigma$):
* **ReLU**: $\max(0, x)$
* **Tanh**: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
* **Sigmoid**: $\sigma(x) = \frac{1}{1 + e^{-x}}$
* **Linear**: $f(x) = x$

---

## 2. Training an MLP Classifier with Dropout & BatchNorm

```swift
import Foundation
import SwiftML

// 1. Synthetic training dataset (100 samples, 4 features)
let sampleCount = 100
let inDim = 4
let X_train = (0..<sampleCount).map { i in
    (0..<inDim).map { d in Double(i * inDim + d) * 0.02 }
}
let y_train = (0..<sampleCount).map { $0 % 2 } // Binary classification

// 2. Configure MLPClassifier with hidden layers, Adam optimizer, Dropout, and BatchNorm
let mlp = try MLPClassifier(
    hiddenLayerSizes: [32, 16],
    activation: .relu,
    learningRate: 0.01,
    maxEpochs: 150,
    batchSize: 16,
    dropoutRate: 0.2,
    batchNorm: true,
    randomSeed: 42
)

// 3. Train model asynchronously under Swift 6 actor isolation
try await mlp.fit(features: X_train, targets: y_train)

// 4. Predict class labels and posterior probabilities
let X_test = [[0.1, 0.2, 0.3, 0.4], [1.5, 1.8, 2.1, 2.4]]
let predictions = try await mlp.predict(features: X_test)
let probabilities = try await mlp.predictProba(features: X_test)

print("Predictions: \(predictions)")
for (idx, prob) in probabilities.enumerated() {
    print("Sample \(idx) Class 1 Probability: \(String(format: "%.2f%%", prob[1] * 100))")
}
```

---

## 3. Training an MLP Regressor

For continuous target estimation:

```swift
let y_reg = X_train.map { $0.reduce(0.0, +) * 2.5 }

let regressor = try MLPRegressor(
    hiddenLayerSizes: [64, 32],
    activation: .tanh,
    learningRate: 0.005,
    maxEpochs: 200,
    batchSize: 32
)

try await regressor.fit(features: X_train, targets: y_reg)
let regPredictions = try await regressor.predict(features: X_test)
print("Regression Predictions: \(regPredictions)")
```

---

## 4. Binary Core ML Model Export

Export trained MLP architectures directly to Apple Core ML binary protobuf `.mlmodel` files:

```swift
try await mlp.writeCoreML(
    to: URL(fileURLWithPath: "TabularMLP.mlmodel"),
    featureNames: ["feat_0", "feat_1", "feat_2", "feat_3"],
    outputName: "target_class"
)
print("Successfully exported MLP to Core ML binary format.")
```
