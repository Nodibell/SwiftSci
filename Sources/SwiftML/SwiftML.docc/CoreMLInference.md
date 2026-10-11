# Run prepared data through Core ML

Use ``CoreMLPredictor`` to run a compiled Core ML regression, vector, or matrix model on
`PreparedNumericBatch`. The predictor matches named scalar features or packs bounded batches
for a vector input. Results retain the input's row order and source indices.

## Export, compile, and predict

Fit preprocessing on training observations. Apply those same fitted transformations
to inference inputs. The predictor does not fit, impute, scale, or choose feature order.

```swift
import CoreML
import SwiftML
import SwiftPreprocessing

// fittedModel and preparedInput use the same trained feature order.
try await fittedModel.writeCoreML(to: modelURL,
    featureNames: ["temperature", "pressure"], outputName: "prediction")
let compiledURL = try await MLModel.compileModel(at: modelURL)
defer { try? FileManager.default.removeItem(at: compiledURL) }
let predictor = try CoreMLPredictor(compiledModelURL: compiledURL,
    inputColumns: ["temperature", "pressure"], outputName: "prediction",
    computeUnits: .all)
let budget = try MemoryBudget(limit: 64 * 1_048_576)
let result = try await predictor.predict(preparedInput, budget: budget,
    workspaceBytes: estimatedModelWorkspaceBytes)
let predictions = result.values
```

Compile once and retain the predictor for repeated requests. The caller owns the source
and compiled artifacts. For persistent applications, keep the compiled model in a stable
location so Core ML can reuse its device-specialization cache.

## Input and precision contract

- Named inputs must be Double scalar features. Their names must exactly match the
  configured columns; physical column order may differ.
- A single vector input takes columns in the explicit `inputColumns` order. The model
  must declare a positive one-dimensional Double or Float32 shape of that width.
  Float16 vectors are also supported on macOS 15 or later.
- Missing values and nonfinite inputs throw. A finite Double that overflows the
  model's Float32 or Float16 element type also throws.
- Output must be a Double scalar or a one-dimensional vector with a supported
  floating-point element type. Vector
  outputs with more than one element become `outputName[0]`, `outputName[1]`, and so on.
  Integer labels are rejected rather than converted to potentially inexact Double values.

Float16 input converts each value to the model's declared type during packing.
This conversion can round finite values or underflow them to zero. Selecting Float16
is a model precision choice, not an automatic memory optimization. It does not
change the configured compute units. On macOS 14, loading a Float16 contract through
this adapter throws an availability error. Double and Float32 remain supported.

Output storage is Double, but this does not imply Double computation inside Core ML.
SwiftSci's neural exporter writes Float32 weights. Core ML may use lower-precision
execution depending on the model and allowed devices. Validate the exported model
against the application's numerical tolerances. A completed prediction that exceeds
a Double-reference tolerance is an accuracy limitation for that requirement; it does
not by itself establish an integration or hardware defect. Choose the permitted devices
and model precision according to the application's accuracy contract.

Images, tensors above rank two, flexible matrix shapes, dynamic output widths, stateful models, and categorical
outputs are outside this adapter's contract. Use Core ML directly for those models.

## Predict a fixed matrix in one call

Choose ``CoreMLInputLayout/matrix`` explicitly when the compiled model declares one
floating-point input shaped `[rows, features]` and output shaped `[rows, outputs]`.
Double and Float32 work on the package's minimum macOS version. Float16 requires
macOS 15 or later. The input and output must have the same positive row count. Feature
columns follow `inputColumns`; physical column order may differ.

```swift
let predictor = try CoreMLPredictor(compiledModelURL: compiledURL,
    inputColumns: ["temperature", "pressure"], outputName: "prediction",
    computeUnits: .all, inputLayout: .matrix)
let result = try await predictor.predict(preparedInput, budget: budget,
    workspaceBytes: estimatedModelWorkspaceBytes)
```

The prepared input must contain exactly the model's declared number of rows. A matrix
request submits all rows together. Leave `maximumBatchSize` at 1; it controls example
batching and is not a matrix row limit. Empty input, different row counts, and flexible
matrix shapes are rejected. The adapter does not pad, truncate, or reshape a model.

One output column uses `outputName`. Multiple columns use `outputName[0]`,
`outputName[1]`, and so on. Results preserve input row order and original row indices,
including selected or repeated rows. Each returned result owns its values.

The default ``CoreMLInputLayout/examples`` preserves existing scalar and vector calls.
A matrix model must declare the intended row semantics; the adapter validates shapes
but cannot infer whether an arbitrary model mixes information across rows.

## Memory and cancellation

The predictor retains immutable prepared columns for the request. Vector inputs use
one reusable `MLMultiArray` by default. Set `maximumBatchSize` to a positive value above
one to submit chunks through Core ML's batch prediction API. Each example in a chunk
owns distinct input storage until that call completes. The last chunk may be smaller.
Core ML decides how to execute the examples; batch submission does not guarantee a
single matrix operation. Measure the model before choosing a batch size.
Core ML output values are copied into owned result columns before their providers are
released. The actor owns the model and serializes its synchronous prediction calls.

Admission includes the input payload, full result payload, up to `min(maximumBatchSize, rowCount)` rows of vector input and
output storage, and `workspaceBytes`. Matrix requests reserve the complete input matrix
and declared output matrix instead of bounded example buffers. Supply a workspace estimate that includes Core ML
allocations, object overhead, and storage padding. This is cooperative admission, not a
process-memory limit. Model loading occurs before prediction admission. Persistent model
storage and results retained after return remain the caller's responsibility.

Matrix packing and extraction use bounded row tiles and typed buffers with Core ML's reported strides.
Packing reads compact column segments directly; extraction writes owned result columns.
Cancellation is checked between tiles and before and after synchronous prediction.
The request retains its reservation until computation and extraction stop.

Cancellation removes a waiting request. During execution, cancellation is checked while packing and between
chunks; a submitted prediction completes before the reservation is released. An error discards
partial results. Autorelease pools drain each chunk's temporary feature objects and
request-level Core ML objects before returning.

`inputPackingBytes` reports cumulative logical bytes written into the vector buffer.
It is zero for scalar inputs. `outputCopyBytes` reports Double bytes written to owned
results. These counters exclude Core ML's internal copies, device transfers, boxing, and
allocations. They do not demonstrate zero-copy access to Neural Engine memory.

## Verify placement and performance

The `computeUnits` setting permits devices. It does not force every operation onto a
particular device. On macOS 14.4 or later, `MLComputePlan` reports anticipated placement;
use Instruments to confirm execution. The adapter itself retains the macOS 14 baseline.

The [Core ML qualification](https://github.com/Nodibell/SwiftSci/tree/main/Benchmarks/CoreMLInference)
compares fixed linear and multilayer workloads through native, MLX, and Core ML paths.
It distinguishes public adapter calls from benchmark-only matrix models. It records
first and warm request timing, Double-reference error, copy counts, resident memory,
admission cleanup, and planned devices. Execution completion, reference accuracy,
and application suitability are separate assessments. Timings remain visible beside
accuracy measurements; the fixtures do not establish trained-model quality or a
production performance baseline.

Apple references: [model ownership](https://developer.apple.com/documentation/coreml/mlmodel),
[compute units](https://developer.apple.com/documentation/coreml/mlcomputeunits), and
[model lifecycle and profiling](https://developer.apple.com/videos/play/wwdc2023/10049/).

## Reuse matrix buffers within a scope

Use ``CoreMLMatrixPool`` when a fixed floating-point matrix model receives repeated
or concurrent requests. Double and Float32 are supported on macOS 14. Float16
requires macOS 15 or later. The pool keeps the model and private input/output buffers alive
inside a scope. Callers receive owned `CoreMLPrediction` results and never manage buffer leases.

```swift
let configuration = try CoreMLMatrixPool.Configuration(
    maximumConcurrentPredictions: 2,
    retainedBytes: estimatedModelAndPoolBytes,
    requestBytes: estimatedPeakBytesPerRequest)
let result = try await CoreMLMatrixPool.withPool(
    compiledModelURL: compiledURL,
    inputColumns: ["temperature", "pressure"], outputName: "prediction",
    computeUnits: .all, budget: budget, configuration: configuration
) { pool in
    try await pool.predict(preparedInput)
}
```

The scope reserves `retainedBytes + maximumConcurrentPredictions * requestBytes` before
loading the model. This entire quota stays reserved while idle. A competing pool waits
for a complete quota, so it cannot retain buffers that consume another admitted pool's
workspace allowance. Start competing pools as siblings. Acquiring another pool inside
an existing scope can still cause caller hold-and-wait if the budget cannot fit both.

Include model-loading peaks, retained model state, page-rounded buffers, and pool overhead
in `retainedBytes`. The pool checks its known buffer capacity before allocating buffers.
Core ML does not expose a complete model-allocation bound, so the remaining estimate is
the caller's responsibility. Underestimation does not become an enforced process-memory cap.

Each request allowance must cover its prepared input payload, owned Double result columns,
a possible fallback Core ML output, framework workspace, and object overhead. The pool
checks known request bytes before queuing. Core ML may decline supplied output backing;
the adapter still validates and copies the returned result. Caller-owned queued inputs
and results retained after scope exit need separate accounting.

Use structured child tasks and await them inside the scope. Scope exit drains admitted
predictions and pending slot acquisitions before releasing capacity. Errors and cancellation
return buffers only after native work completes. A pool reference that escapes the scope
rejects later predictions. Returned results remain valid after the pool closes.

The scoped pool preserves fixed shapes, explicit feature order, missing/nonfinite rejection,
row identity, and owned output semantics. It uses native asynchronous Core ML prediction.
Concurrency is a limit on submissions, not a promise of concurrent hardware execution or
better throughput. The ordinary prediction path converts prepared Double columns to the model's declared
element type on each prediction. IOSurface transport remains experimental.

## Convert once for repeated matrix input

Use ``CoreMLMatrixPool/prepare(_:budget:)`` when several predictions will reuse the
same values. It converts and orders columns once, preserves repeated row selections,
and returns an immutable ``CoreMLPreparedMatrix``. The source batch can then change
or be released without changing the prepared matrix.

```swift
let results = try await CoreMLMatrixPool.withPool(
    compiledModelURL: compiledURL,
    inputColumns: ["temperature", "pressure"], outputName: "prediction",
    computeUnits: .all, budget: poolBudget, configuration: configuration
) { pool in
    let input = try await pool.prepare(preparedInput, budget: inputBudget)
    let first = try await pool.predict(input)
    let second = try await pool.predict(input)
    return [first, second]
}
```

Preparation reserves page-rounded numeric storage and estimated identity metadata before
allocating. `reservedBytes` reports this lifetime allowance. It excludes the source batch,
model, prediction workspace, and returned results. Keep those accounted for separately.
A prepared request does not charge the numeric input a second time against `requestBytes`.
Its request allowance still needs to cover owned results, fallback output, and workspace.

Copies share the prepared owner and one reservation. The last reference frees its numeric
storage before scheduling reservation release on the budget actor. Release is asynchronous.
There is no explicit close operation that could invalidate another caller's reference.
An input may outlive its original pool and work with another pool whose row count, element
type, and ordered column names match.

Preparation can wait for capacity. If input and pool share a budget, leave enough headroom
for both lifetime reservations. Holding a pool quota while waiting for capacity that can
only be returned by closing that same pool causes caller hold-and-wait. A waiting preparation
rechecks the pool after admission and rejects a closed pool without retaining its model.

Each prediction copies the converted bytes into an exclusive slot. It does not repeat
numeric conversion and does not share mutable Core ML input arrays between requests.
`inputPackingBytes` includes this copy. Float16 rounding and overflow rules are the same
as the ordinary path. Preparation costs time and retains additional memory, so use measured
reuse counts for the workload rather than assuming it improves every request.

## Fitted preprocessing

Use <doc:FittedMatrixPreparation> to apply training-fitted imputation and standard scaling directly to private model-input storage. Existing `prepare` and `predict` calls remain available for already transformed batches.

## Bounded sources and consumers

Use <doc:BoundedInference> when admission must cover source loading through ordered result consumption. `CoreMLBatchPipeline` reuses the matrix pool and fitted preparation while limiting the number of loaded but unconsumed batches.
