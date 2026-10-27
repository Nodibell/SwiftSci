# Fit and project prepared data with PCA

Fit preprocessing on training observations, apply the fitted preprocessing to held-out observations, and pass prepared data directly to PCA. PCA returns component scores with feature names and original row positions.

```swift
import SwiftDataFrame
import SwiftPreprocessing
import SwiftCluster

let source = try frame.prepareNumericBatch(["temperature", "pressure", "flow"])
let training = try source.selectingRows(trainingRows)
let query = try source.selectingRows(queryRows)

var scaler = StandardScaler()
try scaler.fit(training)
let scaledTraining = try scaler.transform(consuming: consume training)
let scaledQuery = try scaler.transform(consuming: consume query)

let model = try PCA(nComponents: 2, svdSolver: .full, device: .cpu)
let trainingScores = try await model.fitTransform(scaledTraining)
let queryScores = try await model.transform(scaledQuery)

// Both outputs have columns PC1 and PC2.
// queryScores.originalRowIndices preserves queryRows, including repetitions.
let values = try queryScores.matrix(order: .rowMajor)
```

A model fitted from prepared data requires the same ordered feature names for later prepared inputs. Duplicate feature selections remain valid, but their names and positions must match. A successful nested-row fit clears that name constraint. Nested-row transforms always use positional features.

## Keep an existing matrix layout

You can also fit and project a `PreparedNumericMatrix`. The output keeps its input layout and row provenance while replacing the feature names with `PC1` through `PCk`.

```swift
let matrix = try scaledQuery.matrix(order: .columnMajor, missing: .reject)
let scores = try await model.transform(matrix)
// scores.order == .columnMajor
// scores.originalRowIndices == scaledQuery.originalRowIndices
```

Prepared column inputs pack directly into the centered solver buffer. They do not first construct a matrix or nested rows. CPU fitting needs centered column-major workspace for the existing LAPACK and randomized solvers. Matrix inputs are read in their declared layout; fitting and projection preserve retained input snapshots. Dimensionality reduction allocates new score storage.

## Choose the numerical path

| Choice | Behavior |
| --- | --- |
| `.cpu`, solver `.full` | Double centering and LAPACK divide-and-conquer SVD. |
| `.cpu`, solver `.randomized` | The existing deterministic randomized SVD, with its unchanged sketch and power-iteration defaults. PCA omits unused left-vector back-projection. |
| `.cpu`, solver `.auto` | Randomized when `k < 0.8 * min(rows, columns)`, otherwise full. |
| `.gpu` | Float conversion, MLX centering, and the existing SVD on the CPU stream. Exported state is widened to Double. This is not an FP64 GPU SVD. |
| Device `.auto` or `.ane` | Existing hardware-routing rules apply. `resolvedDevice` reports the last successful fit. |

Projection uses the same ascending-feature Double accumulation for training and held-out observations. Explained variances use `max(1, samples - 1)` normalization. Ratios use total input variance, including omitted components. Randomized fitting is approximate.

## Handle invalid input and model state

Fitting requires at least one observation and feature, rectangular rows, finite values, and a component count no greater than the smaller input dimension. Prepared missing entries are rejected. Exporting them to a matrix as NaNs does not make them valid PCA input. GPU fitting also rejects Double values that overflow Float conversion before calling the native solver.

Prepared projection requires finite inputs and rejects nonfinite scores. A zero-row prepared input with matching feature names returns zero-row score columns. Legacy `fit([])` and `transform([])` retain their empty-input errors. Legacy nested transforms retain NaN propagation. The existing dataframe `fitPCA` convenience method retains its missing-to-zero extraction behavior.

A failed fit leaves the previous fitted state intact. `fitTransform` performs an ordinary successful fit followed by projection. A later projection failure does not roll back that successful fit. Serialize related fit and transform calls on a shared model when they must use one particular fit.

Caller-held inputs, solver workspace, scores, and MLX allocations have separate lifetimes. PCA does not impose an RSS cap. Use the existing memory-admission facilities with an estimate covering the complete operation when your application requires admission control.
