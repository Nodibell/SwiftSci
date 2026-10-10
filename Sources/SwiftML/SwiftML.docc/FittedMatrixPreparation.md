# Prepare model input with fitted preprocessing

Apply training-fitted imputation and standard scaling while filling a Core ML model's input storage.

## Fit on training rows

Create a `StandardPreprocessingPlan` from the training batch. Reuse the plan for held-out and incoming batches. Keep their feature names and order identical to training.

```swift
import SwiftDataFrame
import SwiftPreprocessing
import SwiftML

let plan = try StandardPreprocessingPlan(training: trainingBatch)
```

The default imputation strategy is mean. Pass `strategy:` to select another `Imputer.Strategy`. Scaling uses the existing `StandardScaler` rules after imputation.

## Prepare and predict

Inside an existing `CoreMLMatrixPool.withPool` operation, capture its input preparation value. Call `prepare` from the caller task, then pass the resulting owner to the pool.

```swift
let preparation = try await pool.inputPreparation()
let prepared = try await preparation.prepare(
	incomingBatch,
	preprocessing: plan,
	budget: inputBudget
)
let prediction = try await pool.predict(prepared)
```

The batch must match the model's fixed row count and the plan's column order. Imputation and scaling run in Double precision. Conversion to the model's Float16, Float32, or Double input type happens last. Float16 model input requires macOS 15 or later.

For concurrent requests, share the immutable plan and preparation value. Use bounded child tasks to prepare separate batches. Preprocessing runs outside the pool actor, while the pool limits simultaneous predictions.

## Account for input lifetime

Provide enough input budget for the prepared output and temporary preprocessing buffers together. Preparation reserves both before allocating, then releases the workspace allowance. The returned `CoreMLPreparedMatrix` keeps its output allowance until its last reference is released.

Account for source batches and the fitted plan separately. If the pool and input use the same budget, leave headroom beyond the pool's lifetime quota. An acquisition can wait for earlier owners to release capacity. These allowances do not cap the process allocator or measure resident memory.

A preparation value contains model-input metadata. It does not retain the model, prediction slots, or the pool's quota. It remains usable after the original pool closes. A prepared input can be used by another pool with matching shape, element type, and column order.

The prepared storage is immutable. Prediction still copies its bytes into an exclusive pool slot. Nonfinite or overflowing converted values throw, and cancellation or failure releases the preparation reservation.
