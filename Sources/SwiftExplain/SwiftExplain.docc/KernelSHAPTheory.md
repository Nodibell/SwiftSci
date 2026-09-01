# Game-Theoretic Interpretability with Kernel SHAP

Compute exact and sampled Shapley feature attributions to explain predictions of any machine learning model.

## Overview

**SHAP (SHapley Additive exPlanations)** is a game-theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from cooperative game theory.

## 1. The Four Shapley Axioms

Shapley values are the only attribution method that uniquely satisfies all four fundamental properties:

1. **Efficiency**: Feature attributions sum up to the difference between model prediction `f(x)` and expected background prediction `E[f(X)]`:
   > `Σ_i=1..M φ_i = f(x) - E[f(X)]`
2. **Symmetry**: If two features contribute equally to all possible feature subsets, their attributions are equal.
3. **Dummy (Null Player)**: A feature that does not change the model prediction for any coalition receives an attribution of zero (`φ_i = 0`).
4. **Additivity**: For ensemble models `f = f1 + f2`, attributions satisfy `φ_i(f) = φ_i(f1) + φ_i(f2)`.

## 2. Explaining Predictions in SwiftSci

```swift
import SwiftExplain
import SwiftML

// Initialize KernelSHAP with background dataset
let explainer = try KernelSHAP(
    model: { features in
        // Arbitrary inference closure
        return try await rf.predictProba(features: features).map { $0[1] }
    },
    background: backgroundSamples,
    nSamples: 100
)

// Compute Shapley values for an instance
let instance = [35.0, 75000.0, 720.0, 1.0]
let explanation = try await explainer.explain(instance: instance)

print("Base Value: \(explanation.baseValue)")
print("Feature Attributions: \(explanation.values)")
```

## Topics

### SHAP Explainer APIs
- ``KernelSHAP``
