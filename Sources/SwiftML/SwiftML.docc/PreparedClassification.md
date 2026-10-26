# Train and predict from prepared classification data

Keep features and targets aligned through selection, fit preprocessing on training rows, and predict held-out rows without constructing nested feature arrays.

## Select observations before separating targets

Create one named snapshot with both features and the target. Select training and held-out rows through `PreparedSupervisedBatch` so targets follow the same selection, including repeated rows.

```swift
import SwiftML
import SwiftPreprocessing

let batch = try PreparedNumericBatch(
	columnNames: ["signal", "class"],
	columns: [[-3, -2, 2, 3, -1, 1], [0, 0, 1, 1, 0, 1]])
let observations = try PreparedSupervisedBatch(batch, targetColumn: "class")
let training = try observations.selectingRows([0, 1, 2, 3])
let heldOut = try observations.selectingRows([4, 5])
let classifier: any ClassifierEstimator = ClassificationPipeline(
	transformers: [StandardScaler()],
	estimator: LogisticRegression(device: .cpu))

try await classifier.fit(training)
let probabilities = try await classifier.predictProbability(features: heldOut.features)
let labels = try await classifier.predict(features: heldOut.features)
// labels == [0, 1]
// Output row i belongs to heldOut.features.originalRowIndices[i].
```

Prediction reuses the scaler fitted on the training rows. Do not fit preprocessing on held-out observations. Source mutations preserve the selected snapshots through copy-on-write.

## Choose the number of training updates

Use the concrete logistic estimator for configurable learning rates and epoch counts. Reuse the fitted scaler and transformed batches when the training split and preprocessing settings stay fixed.

```swift
var scaler = StandardScaler()
try scaler.fit(training.features)
let scaledTraining = try scaler.transform(training.features)
let scaledHeldOut = try scaler.transform(heldOut.features)
let model = LogisticRegression(device: .cpu)
try await model.fit(
	features: scaledTraining, targets: training.targets,
	learningRate: 0.125, epochs: 32)
let predictions = try await model.predict(features: scaledHeldOut, threshold: 0.5)
```

Every fit starts from zero parameters. Epochs must be nonnegative; zero epochs produces probabilities of 0.5 and class 0 at threshold 0.5. Labels use a strict `probability > threshold` comparison. This API does not add warm starts or incremental fitting. Use 0/1 labels for binary classification; finite soft targets retain the row API's behavior.

The CPU path keeps Double arithmetic and shares its update and accumulation order with row-array fitting. Prediction and gradient buffers are allocated once per fit and reused across epochs. The GPU path keeps Float MLX training and existing hardware routing. Finite Double inputs can overflow or underflow during Float conversion; choose `.cpu` when that conversion does not meet your numerical requirements. Returned Double probabilities do not restore lost Float precision. Existing GPU finite-step training results also differ from CPU results at the same controls. Prepared integration preserves those GPU updates; CPU and GPU are not interchangeable numerical references.

## Preserve the fitted feature contract

Prepared fitting requires nonempty, uniquely named, finite features and aligned finite targets. Fill missing values before this boundary with preprocessing fitted on training data. After prepared fitting, the native logistic model and pipeline reject changed feature names or order. A row-fitted or loaded model has no stored names and interprets prepared columns by position. The persisted model format continues to store parameters only.

Prepared classifier methods are protocol requirements. Logistic regression and classification pipelines consume prepared inputs natively. Existing classifiers receive compatibility defaults that convert to rows; unsupported probability estimation still throws. Those defaults do not remember a named schema, so use the pipeline when a legacy classifier needs input-name checks. Prepared transformers must preserve row order and identity; the classification pipeline checks that contract after each step.

Configure a pipeline before fitting. Fits reject overlapping fits or active predictions, and predictions reject an active fit. Configuration access uses synchronized snapshots; changing transformers after fitting requires a new fit. A configuration change during fitting prevents that fit from publishing a usable pipeline state. Rejected prepared preconditions preserve the previous fit. Once a refit can mutate transformers, a later failure makes all pipeline prediction routes unavailable until another fit succeeds. A failed or canceled direct logistic fit preserves its previous parameters, names and resolved device together. Concurrent direct fits have no promised completion order.

## Account for retained inputs and outputs

Prepared classifier operations accept the existing `MemoryBudget` and caller-supplied whole-operation `MemoryEstimate`. Include retained inputs, preprocessing intermediates, training workspace, GPU conversions, model state and output arrays. A reservation releases on completion or error; models and returned arrays can outlive it and need separate accounting. Admission does not impose a process-memory limit or choose a device.

GPU inputs use the existing scoped packing and completion boundary. Eligible extents share backing storage; other extents copy. Keep concurrent MLX jobs on independent streams as described in <doc:PreparedRegression>. A memory reservation alone does not provide stream isolation or make mutable pipelines safe for concurrent fitting.
