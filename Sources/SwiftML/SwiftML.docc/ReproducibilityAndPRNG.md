# Reproducibility, Determinism & High-Quality PRNG

Ensure bit-exact experimental reproducibility and statistical robustness across ensemble models and clustering estimators.

## Overview

In machine learning workflows, deterministic reproducibility is vital for auditing, hyperparameter tuning, and regression testing. SwiftSci provides:
1. **Xoshiro256++ Generator**: A modern, high-throughput pseudo-random number generator conforming to standard `RandomNumberGenerator` with period $2^{256} - 1$ passing BigCrush.
2. **`randomState` Parameter Propagation**: Standardized `randomState: Int?` parameters across ensemble estimators (``RandomForestClassifier``, ``RandomForestRegressor``) and clustering models (``KMeans``).
3. **Deterministic Parallel Construction**: Fixed-order tree compilation preserving deterministic tie-breaking even under multi-threaded Swift Concurrency execution (`TaskGroup`).

## 1. Using `randomState` in Random Forests

Pass an integer `randomState` to ensure identical bootstrapping and split selections across runs:

```swift
import SwiftML

let rf = try RandomForestClassifier(
    nEstimators: 50,
    maxDepth: 8,
    randomState: 42
)

try await rf.fit(features: X_train, targets: y_train)
let predictions = try await rf.predict(features: X_test)
```

## 2. Low-Level PRNG with `SeededRandom`

`SwiftPreprocessing` provides `SeededRandom`, backed by Xoshiro256++ and SplitMix64 initialization:

```swift
import SwiftPreprocessing

var rng = SeededRandom(seed: 12345)
let randomValue = rng.nextDouble() // in [0, 1)
let randomIndex = rng.nextInt(upperBound: 100)

// Conforms to Swift's standard RandomNumberGenerator
var items = [1, 2, 3, 4, 5]
items.shuffle(using: &rng)
```

## Topics

### Estimators
- ``RandomForestClassifier``
- ``RandomForestRegressor``
