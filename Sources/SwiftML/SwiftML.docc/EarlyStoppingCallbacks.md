# Early Stopping & Generalization Control

Halt iterative training dynamically when validation performance ceases to improve to prevent overfitting.

## Overview

Iterative estimators like multi-layer perceptrons (`MLPClassifier`, `MLPRegressor`) and gradient boosted decision trees (`GradientBoostedTreesRegressor`) improve their training loss over successive iterations. However, after sufficient capacity has been fit, additional epochs frequently lead to overfitting on training noise and deteriorating generalization on unseen data.

`EarlyStopping` provides a decoupled, value-type training callback that monitors validation loss or performance metrics across successive epochs, halting training and optionally restoring optimal model weights.

## Mechanics & State Machine

```
Epoch Evaluation (Score S_t)
            │
            ▼
 Is (Best - S_t) > minDelta?
   ├── YES ──► Reset wait = 0, update bestScore = S_t, checkpoint weights
   └── NO  ──► Increment wait = wait + 1
                 │
            Is wait >= patience?
              ├── YES ──► shouldStop = true, restore best checkpoint weights
              └── NO  ──► Continue training
```

### Configuration Parameters

- `patience`: Number of consecutive non-improving evaluations tolerated before triggering an early halt.
- `minDelta`: Minimum absolute improvement required to qualify as progress.
- `direction`: `.minimize` for loss functions (cross-entropy, MSE) or `.maximize` for scoring metrics (accuracy, R2, AUC).
- `restoreBestWeights`: When `true`, automatically restores the model parameters from the epoch that achieved `bestScore`, avoiding overfit tail iterations.

## Example Usage

```swift
import SwiftML

// 1. Configure early stopping callback
let es = EarlyStopping(
    patience: 5,
    minDelta: 1e-4,
    direction: .minimize,
    restoreBestWeights: true
)

// 2. Fit MLP with validation monitoring
let mlp = MLPClassifier(hiddenLayerSizes: [32, 16], maxIter: 200)
try await mlp.fit(
    features: X_train,
    targets: y_train,
    validationFeatures: X_val,
    validationTargets: y_val,
    earlyStopping: es
)
```

## Topics

### Training Callbacks
- ``EarlyStopping``
- ``MetricDirection``
