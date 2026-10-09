# Prepared regression

Fit and predict from compact numeric columns without constructing nested row arrays in the StandardScaler and LinearRegression path.

## Keep observations and targets aligned

Create a `PreparedSupervisedBatch` from one named batch before selecting rows. Selection carries features, targets and original row identities together, including repeated rows. Mutating the original batch does not change the selected snapshot.

```swift
import SwiftML
import SwiftPreprocessing

let batch = try PreparedNumericBatch(
    columnNames: ["temperature", "output"],
    columns: [[10, 20, 30, 40], [21, 41, 61, 81]])
let observations = try PreparedSupervisedBatch(batch, targetColumn: "output")
let training = try observations.selectingRows([0, 1, 2])
let heldOut = try observations.selectingRows([3])
let pipeline = RegressionPipeline(
    transformers: [StandardScaler()], estimator: LinearRegression(device: .cpu))
try await pipeline.fit(training)
let predictions = try await pipeline.predict(features: heldOut.features)
```

Fit preprocessing on training rows only. Prediction reuses the learned scaling parameters. The pipeline and LinearRegression remember the named column order after prepared fitting and reject a different order at prediction. A model fitted with unnamed rows cannot infer feature names. Prepared regression requires finite values and aligned finite targets; resolve missing values before fitting.

Configure a pipeline before fitting, and serialize configuration changes and fits. Do not predict during a fit. A rejected prepared-input precondition preserves the previous fitted state. Once a refit starts changing transformers, failure or cancellation makes the pipeline unavailable for prediction until another fit succeeds.

## Compatibility and numerical behavior

Existing row-array calls and conformers remain supported. Prepared calls through `RegressorEstimator` and `PreprocessingTransformer` use protocol requirements, so implementations can consume columns directly. StandardScaler and LinearRegression do this. Default implementations convert to rows for existing conformers. The pipeline supplies feature-name checks for those conformers; their defaults alone do not store a schema.

The transformer compatibility default preserves row count, feature count, names and provenance. A transformer that adds or removes features must implement its prepared overload with the resulting schema. The default rejects a changed shape.

CPU row-array and prepared regression share the same Double QR solver, compensated refinement and gradient fallback. GPU regression still uses Float gradient descent and existing hardware routing. Explicit `.cpu` keeps the CPU numerical path. This API does not change precision or device policy.

StandardScaler retains compact output columns. For prepared batches with at least 32 rows and at most 512 columns, its transform uses a tile capped at 16 KiB and three width-sized temporary arrays. Their combined logical payload is at most 28 KiB, excluding output columns, fitted state and allocation overhead. Smaller or wider inputs use row scratch. Include these allocations in the operation estimate.

## Reuse inputs across fits

Keep a prepared batch when several models use the same observations. This avoids repeated conversion while retaining its value buffers. Include those retained buffers in memory estimates for subsequent work.

When the training split and preprocessing configuration stay fixed, fit the scaler once and reuse its transformed inputs:

```swift
var scaler = StandardScaler()
try scaler.fit(training.features)
let scaledTraining = try scaler.transform(training.features)
let scaledHeldOut = try scaler.transform(heldOut.features)
let model = LinearRegression(device: .cpu)
try await model.fit(features: scaledTraining, targets: training.targets)
let predictions = try await model.predict(features: scaledHeldOut)
```

Further models can use the same scaled batches. Keep the fitted scaler for transforming new observations. Fit it again when training rows, input values, feature order or preprocessing settings change. Do not fit preprocessing on held-out observations or reuse a fitted scaler across different cross-validation training splits.

Caching consumes memory for as long as the caller retains the batches. A one-off fit may gain nothing. For changing splits, preparing the entire source and then gathering each fold retains an extra representation; compare its total time and peak memory with preparing each fold directly. The library does not keep a global prepared-input or scaler-workspace cache.

## Admit estimated peak memory

Share one `MemoryBudget` across participating operations. Supply the peak estimate for the whole operation, including retained input, intermediate data, solver workspace, GPU copies and a runtime allowance. For example, if workload measurements justify a 64 MiB reservation:

```swift
let budget = try MemoryBudget(limit: 256 * 1024 * 1024)
let estimate = try MemoryEstimate(capacities: [64 * 1024 * 1024])
try await pipeline.fit(training, budget: budget, estimate: estimate)
let predictions = try await pipeline.predict(
    features: heldOut.features, budget: budget, estimate: estimate)
```

The budget queues complete reservations in order and releases them on completion or error. Requests larger than the entire budget fail immediately. Cancellation removes queued work without reserving bytes. Qualified GPU consumers retain their input owners until submitted work completes, including error paths.

This is admission control over caller estimates, not a process-memory limit. Model state and returned arrays can outlive a reservation. Account for retained results in subsequent estimates. `WiredMemoryManager` remains a separate task-count limiter.

## Admit independent jobs together

Give each concurrent job its own mutable model and pipeline. Jobs may read the same immutable prepared batch. Retained scaled input is appropriate only when they use the same training split and fitted preprocessing.

Reserve shared input storage once for the entire batch of jobs, including its construction peak if construction occurs inside that reservation. Each job then reserves its additional preprocessing, solver workspace and output estimate. Keep the shared reservation alive until all consumers finish. Account separately for results that remain alive afterward.

The budget must fit the shared reservation and the largest job estimate together. An inner request can otherwise wait indefinitely for bytes held by its outer scope. MemoryBudget does not infer this dependency or account for shared ownership automatically.

Admission limits estimated bytes. It does not select a CPU thread count or guarantee the best throughput. Compare batch completion time, per-job latency and peak memory for the intended workload. Native numerical libraries may also use worker threads. A CPU admission result does not establish a safe or efficient GPU concurrency policy.

### Share a budget across CPU and GPU jobs

For independent MLX consumers, use a private task stream with `MLX.Stream.withNewDefaultStream`. `Device.withDefaultDevice` selects a device but can reuse the existing stream. A memory reservation does not provide stream isolation.

Acquire each job's reservation before allocating its model and workspace. Keep the reservation until its submitted device work completes, including cancellation and error cleanup. For example, with `MLX` imported and `input`, `query` and `targets` already covered by a shared input reservation:

```swift
let predictions = try await budget.withReservation(jobEstimate) {
    try await MLX.Stream.withNewDefaultStream(device: .gpu) {
        defer {
            MLX.Stream.gpu.synchronize()
            MLX.Stream.cpu.synchronize()
        }
        let model = LinearRegression(device: .gpu)
        let pipeline = RegressionPipeline(transformers: [], estimator: model)
        try await pipeline.fit(features: input, targets: targets)
        return try await pipeline.predict(features: query)
    }
}
```

This example expects input scaled using training-only parameters. Each job owns its model. Account separately for predictions or models retained after a reservation ends. Apply stream isolation to CPU jobs that also create MLX compatibility arrays.

Measure MLX's active and cached memory separately from reserved bytes and process RSS. MLX can retain reusable buffers after jobs finish. Increasing admission capacity can increase contention as well as memory use; compare limits on the intended mix of jobs. The CPU and GPU routes use different solvers and precision, so their timings do not establish an interchangeable-device speed ranking.
