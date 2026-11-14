# Fast Tabular Learning with Histogram Gradient Boosting

Discretize continuous feature distributions into up to 256 compact bins (`UInt8`) so tree-node split gain scans take $O(K \cdot p)$ time over $K \le 256$ bins after an $O(N_{\text{node}} \cdot p)$ histogram accumulation pass.

## Overview

Traditional gradient boosted decision trees (GBDT) evaluate continuous threshold candidates by sorting or scanning exact values across all samples at every tree node. When datasets grow large, sorting becomes a major performance bottleneck.

`HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` implement histogram-based split optimization (inspired by LightGBM and scikit-learn's `HistGradientBoostingClassifier`). Continuous feature columns are pre-binned into at most 256 integer bins (`UInt8`).

## Algorithmic Architecture

1. **Pre-Binning (Quantization)**:
   Feature quantiles are evaluated once before boosting begins, establishing up to 256 discrete bins per feature column. All feature values are mapped to compact `UInt8` memory arrays.
2. **One-Pass Histogram Accumulation**:
   For any internal tree node, first-order negative gradients \(g_i\) and second-order Hessians \(h_i\) are accumulated into flat 256-bin histograms in an \(O(N_{\text{node}} \cdot p)\) linear pass over the node's samples:
   $$G[k] = \sum_{i: b_i = k} g_i, \quad H[k] = \sum_{i: b_i = k} h_i$$
3. **Histogram-Bin Split Scan**:
   Once histograms are built, evaluating all candidate thresholds scans only the \(K \le 256\) populated bins per feature (\(O(K \cdot p)\)), rather than re-sorting \(N_{\text{node}}\) continuous values:
   $$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G_{\text{tot}}^2}{H_{\text{tot}} + \lambda} \right]$$

```
Continuous Floats [0.14, 0.92, -0.45] ──► Pre-Binning ──► UInt8 Bins [32, 215, 12]
                                                                  │  O(N_node · p) build
                                                                  ▼
Node Split Scan: O(K · p), K <= 256 ◄─── Accumulated Histograms G[k], H[k]
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
