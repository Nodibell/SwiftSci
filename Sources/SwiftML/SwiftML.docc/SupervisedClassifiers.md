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

### 2. Multi-Layer Perceptron (MLP)

```swift
let mlp = MLPClassifier(hiddenLayerSizes: [64, 32], activation: .relu, maxEpochs: 200)
try await mlp.fit(features: X_train, targets: y_train)
```
