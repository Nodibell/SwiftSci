# Reuse standardized CPU predictions

Fit one preprocessing and model snapshot, then prepare and reuse selected inputs under an explicit memory budget.

## Fit on training data

Configure the existing regressor and call `fitStandardized`. The method fits imputation and standard scaling on training features, trains with those transformed Double values, and returns an immutable `FittedMLPPredictor`.

```swift
import SwiftDataFrame
import SwiftPreprocessing
import SwiftML

let model = MLPRegressor(hiddenLayerSizes: [64, 32], seed: 42)
let fitted = try await model.fitStandardized(
	features: trainingBatch,
	targets: trainingTargets
)
```

Missing entries and numeric NaNs receive the fitted imputation value. Pass `imputationStrategy:` to select another `Imputer.Strategy`. Transformed features, targets, and captured weights must be finite. Incoming feature names and order must match training, including for zero rows.

For early stopping, pass `validation:` as a `PreparedSupervisedBatch` and supply `earlyStopping:`. Validation uses the training-fitted plan and never refits its statistics. Validation rows must be nonempty and have the same feature schema.

The returned predictor owns the weights from this fit invocation. Concurrent or later fits on `model` cannot change them. The mutable actor now holds weights trained in standardized feature coordinates. Its ordinary `predict` methods still expect already-standardized features. Use `fitted` to predict from original features.

Training allocations are outside automatic inference estimates. To admit training, wrap the complete fit in the existing `MemoryBudget.withReservation` with an application-supplied estimate covering training data, preprocessing, conversions, optimizer state, and the returned owner. That estimate does not cap physical allocation. Cancellation is checked before and after synchronous training.

## Record selection and edits

A request retains its source snapshot and supplied selection arrays. It defers identity selection, edited-column gathering, and ordering until preparation admission.

```swift
var request = try MLPPredictionRequest(
	source: incomingBatch,
	selectingRows: selectedOccurrences
)
try request.updateColumn(
	at: featureColumn,
	rows: editedOccurrences,
	values: replacements
)
```

Selection positions refer to the source batch. Edits refer to logical occurrences in that selection, including repeated source rows. Repeated edits use the last supplied value. Nil marks a missing value. Invalid edits throw before changing the request. Copies remain snapshots.

Use `MLPPredictionRequest(source:)` for all rows. A small selection retains the complete source. Only edited columns gather selected values. If releasing unselected source data matters more than avoiding gathers, first create a smaller batch with `PreparedNumericBatch.selectingRows`.

## Prepare and predict

Create a budget from the application's memory policy. Preparation quotes include participating source storage, request arrays, edited columns, ordering, finite-value validation, and workspace.

```swift
let budget = try MemoryBudget(limit: applicationMemoryAllowance)
let preparation = try fitted.preparationEstimate(for: request)
let input = try await fitted.prepare(request, budget: budget)
let prediction = try fitted.predictionEstimate(for: input)
let values = try await fitted.predict(input, budget: budget)
```

A quote larger than the budget's limit throws. Otherwise admission may wait for earlier work to release capacity. Preparation uses one reservation and releases it before returning the immutable input. Prediction includes captured weights, retained input, workspace, scratch, and replacing output buffers. It releases its reservation after computation and output validation.

Results follow original logical occurrence order, including repeated selections and partial batches. Prediction uses CPU Double arithmetic. It does not dispatch to a GPU or Neural Engine. Concurrent calls share immutable owners and receive separate workspace under the shared budget.

An input belongs to the fitted predictor that prepared it. Another predictor rejects it before admission, even with equal schema. A new fit requires new prepared inputs. This prevents accidental pairing of unrelated preprocessing and weights.

## Control retained storage

Keep prepared inputs only when the workload benefits from reuse. The application owns cache lookup, retention limits, invalidation, and eviction. `retainedBytes` on the predictor and inputs provides a conservative capacity charge. Shared arrays may be charged more than once.

Idle owners hold no reservation. Dropping their last references releases storage through Swift ownership. An input retains preprocessing and data, but does not retain weights, the fitted predictor, or the mutable actor. Accumulated output arrays and unrelated owners require separate accounting.

Budget admission is cooperative. It does not intercept allocators or measure total process memory. Do not hold a complete reservation and call an operation that acquires another allowance from the same exhausted budget.

## Handle cancellation and numerical errors

Queued preparation and prediction can cancel before a grant. Once synchronous work starts, cancellation is checked around its documented stages. Individual sorts, training iterations, and BLAS calls are not interrupted midway. Reservations remain held until that work finishes.

Preparation rejects nonfinite standardized incoming values after applying edits and imputation. Prediction rejects nonfinite results. Failure and observed cancellation release active reservations. Inputs and fitted owners remain immutable and can be reused after a cancelled request.
