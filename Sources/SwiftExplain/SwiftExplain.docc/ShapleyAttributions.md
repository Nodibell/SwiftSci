# SHAP & LIME Model Interpretability

Explain machine learning model predictions with game-theoretic Shapley value attributions (TreeSHAP, KernelSHAP) and local linear surrogates (LIME).

## Overview

Black-box machine learning models—such as deep neural networks, random forests, and gradient boosting trees—produce accurate predictions but obscure the individual feature contributions that drove a particular decision.

`SwiftExplain` provides mathematically grounded, axiomatic interpretability algorithms:

1. **``TreeSHAP``**: Fast, exact polynomial-time Shapley attributions for tree-based models (`DecisionTreeClassifier`, `RandomForestClassifier`, `RandomForestRegressor`) adhering to the four Shapley efficiency and consistency axioms.
2. **``KernelSHAP``**: Model-agnostic explainer solving constrained weighted linear regression against an empirical background dataset for any arbitrary asynchronous prediction function.
3. **``LIMEExplainer``**: Local Interpretable Model-agnostic Explanations fitting ridge linear surrogate models around perturbed neighborhoods.

```mermaid
graph TD
    Model[Trained Model / Ensemble] --> ModelType{Model Architecture}
    ModelType -->|Tree Ensemble / Forest| Tree[TreeSHAP: Exact Lundberg Algorithm]
    ModelType -->|Arbitrary Black-Box / Pipeline| Kernel[KernelSHAP: Empirical Regression]
    ModelType -->|Local Linear Surrogate| LIME[LIMEExplainer: Perturbation Neighborhood]
    Tree --> Attributions[Exact Shapley Values: sum phi_i = f(x) - E(f)]
    Kernel --> Attributions
    LIME --> SurrogateWeights[Local Coefficients + R^2 Goodness of Fit]
    Attributions --> FeatureRanking[Feature Importance Ranking & Waterfall Breakdown]
```

---

## 1. Exact TreeSHAP for Tree Ensembles

TreeSHAP (Lundberg et al., 2018; Nature Machine Intelligence) optimizes exponential subset evaluation into polynomial time $O(T L D^2)$ by recursively tracking conditional expectations along decision tree splitting paths.

```swift
import SwiftML
import SwiftExplain

// 1. Train a Random Forest model on feature data
let rf = RandomForestClassifier(nEstimators: 50, maxDepth: 6)
try await rf.fit(features: X_train, targets: y_train)

// 2. Initialize TreeSHAP explainer
let treeShap = TreeSHAP()

// 3. Compute exact Shapley attributions for an individual observation
let testInstance = X_test[0]
let shapValues = await treeShap.explain(randomForest: rf, instance: testInstance)

// 4. Rank features by absolute impact on the decision
let featureNames = ["Income", "DebtToIncome", "CreditScore", "EmploymentYears", "LoanAmount"]
let ranked = zip(featureNames, shapValues).sorted { abs($0.1) > abs($1.1) }

print("Feature attributions for instance:")
for (name, phi) in ranked {
    let direction = phi >= 0 ? "+" : "-"
    print("  • \(name): \(direction)\(String(format: "%.4f", abs(phi)))")
}
```

### Concurrent Batch Explanations

For dataset-wide interpretability or global summary plots, compute attributions concurrently across Apple Silicon CPU cores:

```swift
let flatTrees = await rf.flatTrees
let batchAttributions = await treeShap.explainBatch(
    trees: flatTrees,
    instances: X_test,
    numFeatures: featureNames.count
)
print("Computed SHAP matrix for \(batchAttributions.count) instances.")
```

---

## 2. Model-Agnostic Explanations with `KernelSHAP`

When interpreting models without tree structures—such as support vector machines, neural decoders, or multi-stage pipelines—``KernelSHAP`` estimates Shapley values through weighted linear regression over coalition permutations:

```swift
import SwiftExplain

// 1. Initialize KernelSHAP explainer actor
let kernelShap = KernelSHAP()

// 2. Compute Shapley attributions using background baseline dataset
let attributions = await kernelShap.explain(
    model: { features in
        // Any asynchronous prediction function mapping [Double] -> Double
        return await myNeuralNetwork.predict(features)
    },
    instance: testInstance,
    background: backgroundSampleData,
    numCoalitions: 300
)

print("KernelSHAP Attributions: \(attributions)")
```

---

## 3. Local Surrogate Explanations with LIME

``LIMEExplainer`` perturbs the target instance with Gaussian noise, queries the black-box model on the neighborhood, and trains an interpretable, locally-weighted ridge linear surrogate:

```swift
import SwiftExplain

let lime = LIMEExplainer(kernelWidth: 0.75, regularization: 1.0)

let explanation = await lime.explain(
    model: { x in await complexPipeline.predictScore(x) },
    instance: testInstance,
    numSamples: 500
)

print("Local Model Intercept: \(explanation.intercept)")
print("Local Surrogate Fidelity (R²): \(String(format: "%.3f", explanation.localR2))")

for (idx, weight) in explanation.featureWeights.enumerated() {
    print("Feature \(idx) Local Weight: \(String(format: "%.4f", weight))")
}
```

---

## Topics

### Explainability Engines
- ``TreeSHAP``
- ``TreeSHAPExplanation``
- ``KernelSHAP``
- ``LIMEExplainer``
