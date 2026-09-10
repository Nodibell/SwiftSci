# Fast Tabular Learning with Histogram Gradient Boosting

Discretize continuous feature distributions into 256 compact bins to accelerate decision tree split evaluations from \(O(N \log N)\) to \(O(K)\).

## Overview

Traditional gradient boosted decision trees (GBDT) sort continuous feature values across all samples at every internal tree node to find optimal split thresholds. When datasets grow to millions of rows, sorting becomes the primary performance bottleneck.

`HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` implement histogram-based split optimization (inspired by LightGBM and scikit-learn's `HistGradientBoostingClassifier`). Continuous feature columns are discretized into at most 256 integer bins (`UInt8`).

## Algorithmic Architecture

1. **Pre-Binning (Quantization)**:
   Feature quantiles are evaluated once before boosting begins, establishing up to 256 discrete bins per feature column. All feature values are mapped to compact `UInt8` memory arrays.
2. **One-Pass Histogram Accumulation**:
   For any internal tree node, first-order negative gradients \(g_i\) and second-order Hessians \(h_i\) are accumulated into a flat 256-bin histogram in a single linear pass over the node's samples:
   $$G[k] = \sum_{i: b_i = k} g_i, \quad H[k] = \sum_{i: b_i = k} h_i$$
3. **Constant-Time Split Finding**:
   Evaluating all split candidates requires iterating over \(K \le 256\) bins, regardless of how many millions of samples reside in the node:
   $$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G_{\text{tot}}^2}{H_{\text{tot}} + \lambda} \right]$$

```
Continuous Floats [0.14, 0.92, -0.45] ──► Quantization ──► UInt8 Bins [32, 215, 12]
                                                                  │
                                                                  ▼
Node Split Search: O(256) Bin Scan ◄─── Accumulated Histograms G[k], H[k]
```

## Example Usage

```swift
import SwiftML

let clf = try HistGradientBoostingClassifier(
    nEstimators: 50,
    learningRate: 0.1,
    maxDepth: 6,
    maxBins: 256,
    minSamplesLeaf: 10
)

try await clf.fit(features: X_train, targets: y_train)
let predictions = try await clf.predict(features: X_test)
let probabilities = try await clf.predictProbability(features: X_test)
```

## Topics

### Estimators
- ``HistGradientBoostingClassifier``
- ``HistGradientBoostingRegressor``
- ``HistTreeNode``
