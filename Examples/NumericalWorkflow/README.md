# Train and reuse a numerical workflow

Train a CSV classifier, save its fitted preprocessing and model, then predict
unlabeled rows or evaluate new labeled data in a separate process. This example uses `SwiftWorkflow` to compose
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
- `report.json`: schema version 2, training and query row selections, accuracy
  and the selected training options.

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

This check also covers custom feature/target names, configurable classifier
updates, reordered CSV columns, invalid options/artifacts and output path
protection. A deliberately changed coefficient must fail the independent verifier.
Saved-model evaluation checks an imperfect prediction fixture against independent
confusion-matrix and macro/weighted F1 values, including single-row/single-class data.

## Predict unlabeled rows

Provide a CSV with the model's saved numeric feature names. For the default
sample, these are `signal`, `companion` and `constant`. A target column is
unnecessary. Physical CSV column order may
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

## Evaluate a saved model on labeled data

Provide the saved feature columns and a ground-truth column containing 0 or 1:

```sh
.build-native/Build/Products/Release/ScientificWorkflow evaluate \
  --model .build/my-results/model.json --input /path/to/labeled.csv \
  --target outcome --output .build/evaluation
```

`--target` defaults to `class`. Use the column name in the evaluation CSV; it
cannot also be a saved feature. Evaluation uses every supplied row without
fitting or changing the saved model. To assess generalization, supply data that
was not used for training. A single labeled row is sufficient to run this command.

The output directory contains:

- `predictions.csv`: source-row positions, actual labels, predicted labels and
  both class probabilities, in evaluation-input order.
- `evaluation.json`: schema version 1, target column, source-row positions and
  the existing `SwiftOptimize.ClassificationEvaluation` result. It includes
  accuracy, macro precision/recall/F1, support-weighted F1 and a confusion matrix.

`classLabels` lists the sorted labels observed in ground truth or predictions.
The confusion matrix uses that order for actual-label rows and predicted-label
columns. Macro metrics give each observed class equal weight; weighted F1 uses
ground-truth class frequencies. For single-class data, the matrix may be 1 by 1.

Missing, nonnumeric or nonbinary labels fail before result files are written.
The command rejects output paths that would overwrite its input or saved model.

## Train another dataset

```sh
.build-native/Build/Products/Release/ScientificWorkflow train \
  --input /path/to/data.csv --output .build/my-results \
  --feature temperature --feature pressure --feature humidity --target outcome \
  --components 2 --learning-rate 0.125 --epochs 64
```

Repeat `--feature` for each numeric input column in fitted order. Use quotes for
names containing spaces or commas, such as `--feature "pressure,kPa"`. Other CSV
columns may contain text and remain unused. The target must be separate from all
selected features. Without these options, training uses `signal`, `companion`,
`constant` and target `class`, preserving the bundled example's defaults.

Provide at least six data rows. Each target value must be 0 or 1, and the first
two thirds of the rows must contain both classes. This command supports binary
classification, with the remaining rows held out in reverse order.
Each feature needs a numeric value in the reader's first 1,000 data rows for type
inference. Entirely missing feature columns are rejected before imputation.
The same inference requirement applies to prediction CSVs.

`--components` defaults to 2 and cannot exceed the feature count or training row
count. For a single feature, use `--components 1`. `--learning-rate` must be finite
and positive. `--epochs` must be nonnegative; zero saves a classifier initialized
with zero weights and bias. The report records the settings actually used.
Prediction reads the saved feature names and requires no repeated training options.

Training replaces `model.json`, `predictions.csv` and `report.json` in the chosen
directory. Prediction replaces its output CSV. Use `ScientificWorkflow --help`,
`train --help`, `predict --help` or `evaluate --help` for options.

To train from prepared data in your application, use the library API:

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
