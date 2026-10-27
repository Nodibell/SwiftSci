# Run a complete numerical workflow

This example reads CSV data, fits preprocessing and a classifier on training rows,
then exports held-out predictions with their original row positions. It uses the
public APIs in SwiftDataFrame, SwiftPreprocessing, SwiftCluster, SwiftML and
SwiftOptimize. ArgumentParser provides the command-line interface.

## Run the bundled sample

Use Apple silicon macOS 14 or later with Xcode selected. From the repository root:

```sh
xcodebuild -scheme ScientificWorkflow -configuration Release \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath .build-native \
  MTL_FAST_MATH=NO MTL_MATH_MODE=SAFE MTL_MATH_FP32_FUNCTIONS=PRECISE build
.build-native/Build/Products/Release/ScientificWorkflow --output .build/numerical-workflow
python3 Examples/NumericalWorkflow/verify.py .build/numerical-workflow
```

The executable prints the training/test counts, held-out accuracy and two output
paths. The verifier checks the bundled sample independently and prints a JSON
result. Python is needed only for this optional verification step.

The sample's first output lines are:

```text
Training rows: 8; held-out rows: 4
Held-out accuracy: 1.0
```

Use Xcode for the final build because command-line SwiftPM does not compile the
linked MLX dependency's Metal shaders. The dependency also needs an accessible
Metal environment even though this example requests CPU execution. A headless or restricted launch can fail
before the workflow starts. Normal CI builds compile the example; full numerical
execution is qualified on an Apple silicon Mac with Metal access.

## Follow the stages

1. Read `Data/sample.csv` with compact Double storage. The named features are
   `signal`, `companion` and `constant`; `class` is a separate binary target.
2. Select features and targets together using `PreparedSupervisedBatch`. The first
   two thirds of the rows train the model. The remaining rows are queried in
   reverse order, making source-row alignment visible.
3. Fit `Imputer` on training features and reuse its mean replacements for both
   sets. The sample contains one missing cell and one numeric NaN. Both receive
   the fitted replacement, while the original source snapshot stays unchanged.
4. Fit `StandardScaler` on imputed training features. Reuse its parameters for
   held-out features, including the sample's constant column.
5. Fit two-component CPU `PCA` on scaled training features. Transform held-out
   features using that fitted state.
6. Fit CPU `LogisticRegression` on prepared training scores and their selected
   targets, using 64 updates at learning rate 0.125. Predict labels and both class
   probabilities from the held-out score batch.
7. Associate the returned arrays with `queryScores.originalRowIndices`, evaluate
   `Metrics.accuracy` and export a DataFrame with `writeCSV`.

No learned stage is fitted on held-out rows, and the target never enters feature
preprocessing. The PCA-to-classifier handoff stays prepared. Diagnostic row
exports are made afterwards for the JSON stage report.

## Inspect the results

`predictions.csv` contains `source_row`, `actual`, `predicted`,
`probability_class_0` and `probability_class_1`. Source rows are zero-based data-row
positions, excluding the header. They are positions in this input snapshot,
not business identifiers. The bundled sample has eight training rows and four
held-out rows, exported in source order 11, 10, 9, 8.

`report.json` contains fitted imputation/scaling parameters, PCA components and
variance, stage values, classifier weights, probabilities, row selections and
accuracy. JSON represents missing values and NaNs as `null` in the diagnostic
raw rows; separate counters retain their distinction.

The verifier independently computes training-only preprocessing, a closed-form
two-feature PCA eigensystem and scalar logistic updates. It checks both output
files with `abs(error) <= 1e-12 + 1e-12 * abs(reference)`, allowing PCA sign changes.
It validates this bundled fixture, not arbitrary replacement datasets.

## Use another CSV

```sh
.build-native/Build/Products/Release/ScientificWorkflow --input /path/to/data.csv --output .build/my-results
```

Provide the same four named numeric columns and at least six data rows. Every
`class` value must be 0 or 1 and the training portion must contain both classes.
Each feature must have a numeric value within the CSV reader's first 1,000 data
rows so type inference can establish its numeric type. Entirely missing feature
columns, or columns whose first numeric value appears after that sample, are
rejected before imputation. Missing or NaN cells in numeric columns are imputed;
infinite features, invalid text and invalid labels need correction in the input. Change the feature names, row selection, PCA width or
training settings in `ScientificWorkflow.swift` for a different problem.

Each successful run replaces only `predictions.csv` and `report.json` in the
chosen directory. Use `.build-native/Build/Products/Release/ScientificWorkflow --help` for the CLI options.

The small sample and ordered split demonstrate integration. Their accuracy does
not measure generalization on a representative dataset. This example makes no
speed, accelerator-placement or memory-bound claim; the stage report retains
additional diagnostic arrays.
