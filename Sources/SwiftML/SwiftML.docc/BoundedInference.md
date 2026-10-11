# Run bounded batch inference

Use `CoreMLBatchPipeline.run` to pull batches through fitted preprocessing and a fixed-shape Core ML model. The pipeline holds capacity until each result consumer returns.

## Fit preprocessing on training data

Create a `StandardPreprocessingPlan` from training rows before starting inference. Reuse that plan for all incoming batches. See <doc:FittedMatrixPreparation> for fitting and numeric conversion rules.

Training allocations occur before the pipeline scope. Include the retained plan and source cursor in the run allowance. Account separately for any already loaded dataset.

## Set memory and concurrency allowances

Create a pool configuration and a batch configuration. The following capacities are illustrative. Measure the model and source workload before using them in an application.

```swift
let pool = try CoreMLMatrixPool.Configuration(
    maximumConcurrentPredictions: 1,
    retainedBytes: 32 * 1024 * 1024,
    requestBytes: 16 * 1024 * 1024
)
let configuration = try CoreMLBatchPipeline.Configuration(
    pool: pool,
    maximumInFlightBatches: 2,
    retainedBytes: 1024 * 1024,
    batchBytes: 32 * 1024 * 1024
)
let budget = try MemoryBudget(limit: configuration.requiredBytes)
```

`maximumInFlightBatches` includes loaded sources, active predictions, completed outputs, and the result currently being consumed. It must be at least the pool's prediction-slot count.

The pipeline acquires one reservation before loading the model or invoking the source. Its size is the pool quota plus `retainedBytes` plus `maximumInFlightBatches * batchBytes`.

Include loading buffers, numeric source capacity, prepared storage, preprocessing workspace, owned output, and consumer workspace in `batchBytes`. Include the fitted plan, source state, consumer state, and scheduling overhead in `retainedBytes`. The implementation checks known storage minima. These estimates do not cap Core ML allocations or process residency.

## Supply a source and consumer

Pass closures that load one batch and await its result processing. In this example, `source` and `sink` are application-owned adapters.

```swift
try await CoreMLBatchPipeline.run(
    compiledModelURL: compiledModelURL,
    inputColumns: plan.columnNames,
    outputName: "result",
    computeUnits: .cpuAndNeuralEngine,
    preprocessing: plan,
    budget: budget,
    configuration: configuration,
    load: { request in
        try await source.readBatch(
            sequence: request.sequence,
            rows: request.rowCount,
            columns: request.columnNames
        )
    },
    consume: { sequence, prediction in
        try await sink.write(sequence: sequence, values: prediction.values)
    }
)
```

The source runs serially. Return `nil` at exhaustion. Return a batch with exactly the requested row count and column order otherwise. An incomplete final batch throws. The pipeline does not pad, truncate, repartition, or discard rows. Models can compute across rows, so padding is not generally equivalent.

The consumer also runs serially, in source order. Its sequence number starts at zero. Prediction preserves the batch's `originalRowIndices`, including repeated and reordered rows. Separately loaded chunks usually have local row indices. Use the sequence number and your source mapping to identify their records globally.

Await the consumer's write or reduction before returning. Starting an unstructured write lets it outlive the reserved window. If you intentionally retain an output beyond the callback, provide separate lifetime accounting for it.

## Handle cancellation and errors

Make the source and consumer cooperate with task cancellation. On an observed error, the pipeline stops pulling data, cancels sibling tasks, and awaits submitted work before releasing its reservation. Core ML may finish an already submitted prediction before shutdown completes.

Earlier consumer writes remain committed when a later batch fails. The pipeline supplies no transaction or retry policy. Use idempotent writes or an application transaction if the source may restart.

Use <doc:CoreMLInference> directly when the application already controls source admission and result lifetime, or when it needs to reuse one prepared matrix across predictions.
