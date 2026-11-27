# Model Interpretability & XAI Guide

Understand black-box machine learning predictions using `TreeSHAP`, `KernelSHAP`, `LIMEExplainer`, `PermutationImportance`, and `PartialDependencePlot`.

## Overview

Explainable Artificial Intelligence (XAI) bridges the gap between predictive accuracy and decision transparency. `SwiftExplain` provides exact Shapley mathematical decomposition for tree-based models, local linear surrogate modeling with LIME, and model-agnostic permutation metrics.

---

## 📊 Explainability Methods Comparison

| Method | Supported Models | Algorithmic Complexity | Scope | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **`TreeSHAP`** | `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostedTreesRegressor` | `O(T · 2^K · L)` exact (`K ≤ 12`); `O(T · S · K · L)` seeded MC (`K > 12`, `S = 2048`) | Local & Global | **Exact for `K ≤ 12` distinct split features** using either `.uniformLeaf` (default) or `.empiricalCover` leaf weights; supports GBDT additive explanations and per-class probability decomposition. |
| **`KernelSHAP`** | Any black-box model (MLP, SVM, Ensembles) | `O(S · D²)` | Local | Universal Shapley attribution using weighted coalition sampling. |
| **`LIME`** | Any black-box model (MLP, Vision, NLP) | `O(S · D)` | Local | **Fast local surrogate.** Fits an interpretable weighted linear model around perturbations. |
| **`PermutationImportance`** | Any model | `O(D · N)` | Global | Fast global feature ranking by measuring validation loss degradation upon feature column shuffling. |
| **`PartialDependencePlot`** | Any model | `O(G · N)` | Global | Visualizes marginal effect of 1 or 2 features across an evaluation grid. |

---

## 1. Path-Dependent TreeSHAP for Trees, Forests & GBDTs

`TreeSHAP` evaluates path-dependent conditional expectations over each tree's leaves using either ``TreeNodeWeighting/uniformLeaf`` (default, leaf weight `1.0`) or ``TreeNodeWeighting/empiricalCover`` (leaf weight `FlatTreeNode.cover`, with automatic fallback to `1.0` when decoding legacy `<= 3.13.3` serialized models). When a tree splits on `K ≤ TreeSHAP.maxExactFeatures` (`12`) distinct features, `TreeSHAP` enumerates all `2^K` coalitions in `O(2^K · L)` time to compute exact Shapley values; when `K > 12`, it switches to deterministic seeded Monte Carlo permutation sampling (`approximationSamples = 2048`):

- **GBDT (`GradientBoostedTreesRegressor`)**: `explainDetailed(gbdt:instance:)`, `explain(gbdt:instance:)`, and `explainBatch(gbdt:instances:)` apply `initialPrediction + learningRate * sum_m T_m(x)` with per-tree Shapley attributions scaled by `learningRate`.
- **Per-Class Classification (`DecisionTreeClassifier`, `RandomForestClassifier`)**: `explainClassDetailed`, `explainAllClassesDetailed`, and `explainBatchClass` decompose calibrated `predictProbability` outputs (supporting `ForestProbabilityMode.hardVote` and `ForestProbabilityMode.softLeafDistribution`), while `explain(decisionTree:)` and `explain(randomForest:)` preserve scalar `FlatTreeNode.value` compatibility. `DecisionTreeClassifier.prune(alpha:)` preserves both empirical `classDistribution` and `cover` on collapsed leaves.

```swift
import Foundation
import SwiftML
import SwiftExplain

// 1. Train a Random Forest Classifier
let X = [[1.0, 2.0], [2.0, 1.0], [5.0, 4.0], [6.0, 5.0]]
let y = [0, 0, 1, 1]

let forest = try RandomForestClassifier(nEstimators: 10, maxDepth: 4)
try await forest.fit(features: X, labels: y)

// 2. Explain prediction on a target instance
let treeSHAP = TreeSHAP()
let instance = [5.5, 4.2]

let shapValues = await treeSHAP.explain(randomForest: forest, instance: instance)

print("=== TreeSHAP Feature Attributions ===")
for (featureIdx, attribution) in shapValues.enumerated() {
    print("Feature \(featureIdx): Contribution = \(String(format: "%+.4f", attribution))")
}
```

---

## 2. Local Interpretable Model-agnostic Explanations (`LIME`)

`LIMEExplainer` perturbs the neighborhood of a target sample using Box-Muller Gaussian noise, weights samples by an exponential distance kernel, and fits a weighted Ridge regression surrogate model:

```swift
import SwiftExplain
import SwiftML

// Black-box model closure
let modelClosure: @Sendable ([Double]) async -> Double = { x in
    // Non-linear complex interaction
    return 3.0 * x[0] + sin(x[1] * 2.0) - 1.5 * x[2] * x[2] + 4.0
}

let instance = [1.5, 0.8, -0.4]

let lime = LIMEExplainer(kernelWidth: 0.75, regularization: 0.1)
let explanation = await lime.explain(
    model: modelClosure,
    instance: instance,
    numSamples: 500
)

print("=== LIME Local Surrogate Explanation ===")
print("Local Prediction: \(String(format: "%.4f", explanation.prediction))")
print("Surrogate Intercept: \(String(format: "%.4f", explanation.intercept))")
print("Local Goodness-of-Fit (R²): \(String(format: "%.2f%%", explanation.localR2 * 100))")

for (d, weight) in explanation.featureWeights.enumerated() {
    print("Feature [\(d)] Local Slope: \(String(format: "%+.4f", weight))")
}
```

---

## 3. Global Permutation Feature Importance

Measure global model dependence by quantifying score drop when a single feature column is randomly permuted:

```swift
import SwiftExplain

let perm = PermutationImportance()
let importances = try await perm.computeImportance(
    features: X,
    targets: y.map(Double.init)
) { matrix in
    try await forest.predict(features: matrix).map(Double.init)
}

print("=== Global Feature Importance Ranking ===")
for (feat, score) in importances.sorted(by: { $0.value > $1.value }) {
    print("\(feat): Importance Score = \(String(format: "%.4f", score))")
}
```
