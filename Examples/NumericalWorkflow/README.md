# Train and reuse a numerical workflow

Train a CSV classifier, save its fitted preprocessing and model, then predict
unlabeled rows in a separate process. This example uses `SwiftWorkflow` to compose
mean imputation, standard scaling, CPU PCA and CPU logistic regression.

## Build the executable

Use Apple silicon macOS 14 or later with Xcode selected. From the repository root:

```sh
xcodebuild -scheme ScientificWorkflow -configuration Release \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath .build-native \
  MTL_FAST_MATH=NO MTL_MATH_MODE=SAFE MTL_MATH_FP32_FUNCTIONS=PRECISE build
```

Xcode compiles the linked MLX dependency's Metal shaders. If Xcode reports a
missing Metal Toolchain, install that component with
`xcodebuild -downloadComponent MetalToolchain`. Execution needs an accessible
Metal environment even though this workflow requests CPU computation.

## Train and save the bundled sample

```sh
.build-native/Build/Products/Release/ScientificWorkflow train --output .build/numerical-workflow
python3 Examples/NumericalWorkflow/verify.py .build/numerical-workflow
```

Omitting `train` preserves the original command syntax. The sample prints:

```text
Training rows: 8; held-out rows: 4
Held-out accuracy: 1.0
```

The output directory contains:

- `model.json`: version-1 fitted parameters and ordered feature names. It contains
  no training rows or targets.
- `predictions.csv`: `source_row`, `actual`, `predicted`, `probability_class_0`
  and `probability_class_1` for held-out rows.
- `report.json`: schema version 2, training and query row selections, and accuracy.

The first two thirds of the rows train every learned stage. The remaining rows
are queried in reverse order, making row alignment visible. Missing cells and
numeric NaNs receive training-only mean replacements. The constant feature uses
the scaler's fitted deviation of 1. The classifier uses two PCA components and
64 updates at learning rate 0.125. The target never enters preprocessing.

The verifier independently computes the bundled sample's preprocessing, PCA and
logistic updates. It checks saved parameters and CSV predictions with
`abs(error) <= 1e-12 + 1e-12 * abs(reference)`, allowing PCA sign changes.
Python is needed only for this optional verification.

To exercise saving and prediction across separate processes, run:

```sh
python3 Examples/NumericalWorkflow/verify-persistence.py .build-native/Build/Products/Release/ScientificWorkflow
```

This check also covers reordered CSV columns, invalid artifacts, output path
protection and a deliberately changed coefficient that the verifier must reject.

## Predict unlabeled rows

Provide a CSV with the saved numeric feature names, `signal`, `companion` and
`constant`. The `class` column is unnecessary. Physical CSV column order may
change because the command selects columns by their saved names.

```sh
.build-native/Build/Products/Release/ScientificWorkflow predict \
  --model .build/numerical-workflow/model.json \
  --input /path/to/unlabeled.csv --output .build/new-predictions.csv
```

Prediction loads the saved parameters without fitting any stage. The output
contains the four prediction columns above, without `actual`. `source_row` is the
zero-based data-row position in this prediction input, excluding its header.
These positions are not business identifiers. The library also preserves row
selections, their order and duplicate row IDs in prepared batches.

The loader rejects unsupported artifact versions, invalid parameter values and
incompatible stage dimensions. Prediction rejects missing features, nonnumeric
columns and infinity. Numeric missing values and NaNs use the saved replacements.
Input and model files cannot be chosen as output paths.

## Train another dataset

```sh
.build-native/Build/Products/Release/ScientificWorkflow train \
  --input /path/to/data.csv --output .build/my-results
```

Provide the same four named numeric columns and at least six data rows. Each
`class` value must be 0 or 1, and the training portion must contain both classes.
Each feature needs a numeric value in the reader's first 1,000 data rows for type
inference. Entirely missing feature columns are rejected before imputation.
The same inference requirement applies to prediction CSVs.

Training replaces `model.json`, `predictions.csv` and `report.json` in the chosen
directory. Prediction replaces its output CSV. Use `ScientificWorkflow --help`,
`train --help` or `predict --help` for options.

For other named features or training settings, use the library API:

```swift
import SwiftWorkflow

let fitted = try await PCAClassificationWorkflow.train(training: trainingBatch)
try fitted.save(to: modelURL)
let loaded = try PCAClassificationWorkflow.load(from: modelURL)
let predictions = try await loaded.predict(queryFeatures)
```

`trainingBatch` is a `PreparedSupervisedBatch`. `queryFeatures` is a
`PreparedNumericBatch` whose names match `loaded.featureNames` in order.
Published workflows expose no mutable stages. Saves contain one captured set of
fitted parameters, and prediction never changes that set.

The bundled sample and ordered split demonstrate integration. Their accuracy
does not measure generalization on a representative dataset.
