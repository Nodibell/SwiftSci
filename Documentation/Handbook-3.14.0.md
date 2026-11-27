---
title: SwiftSci 3.14.0 Engineering Handbook
subtitle: Scientific computing and machine learning on Apple Silicon
lang: en
---

The API baseline is [SwiftSci v3.14.0](https://github.com/Nodibell/SwiftSci/tree/release/3.14.0) (branch `release/3.14.0`). The downloadable editions record their exact documentation revision separately.

# Install and select modules

Use macOS 14 or later on Apple Silicon and Swift 6.3 or later. The GPU-backed paths require Metal. Pin the package when a workflow must reproduce this release:

```swift
.package(url: "https://github.com/Nodibell/SwiftSci.git", exact: "3.14.0")
```

Add only the products your target imports, for example `.product(name: "SwiftForecast", package: "SwiftSci")`. Keep the consumer's resolved dependency file and toolchain version with reproducible experiments. See the [package manifest](https://github.com/Nodibell/SwiftSci/blob/release/3.14.0/Package.swift).

| Product | Use it for |
|---|---|
| SwiftDataFrame | Typed tables, joins, grouping, CSV and Parquet |
| SwiftStats | Descriptive statistics, distributions, correlations and tests |
| SwiftPreprocessing | Scaling, encoding, imputation and transforms |
| SwiftML | Linear models, trees, ensembles, neural networks, calibration and model export |
| SwiftCluster | PCA, clustering and approximate nearest-neighbor search |
| SwiftNLP | Tokenization, text features, embeddings and sentiment |
| SwiftOptimize | Model scoring, cross-validation and hyperparameter search |
| SwiftForecast | ARIMA, SARIMA, smoothing, error metrics and backtesting |
| SwiftLLM | Local model loading, inference and MLX execution |
| SwiftExplain | Attribution and local explanations |
| SwiftVisualization | Interactive, HTML and terminal charts |
| SwiftVision | Classification, detection and segmentation |
| SwiftDatabase | SQLite and supported PostgreSQL and MySQL workflows |
| SwiftAgent | Tool interfaces, reasoning loops and retrieval memory |
| SwiftWorkflow | Persisted preprocessing, PCA and classification |

The [online API reference](https://nodibell.github.io/SwiftSci/) links the module guides (`100.00%`, `2,641/2,641` matched public declarations documented across `15/15` DocC archives).

# Typed tables and sample statistics

Choose concrete column types when ingesting data. Distinguish missing values, empty strings and quoted literal null tokens. Keep feature and target identities until the matrix boundary.

```swift
import SwiftDataFrame
import SwiftStats

let x = TypedColumn<Double>(name: "feature", values: [1, 2, 3, 4])
let y = TypedColumn<Double>(name: "target", values: [2, 4, 6, 8])
let frame = try DataFrame(columns: [x, y])
let target = try frame.toTargetVector("target")
print(try Stats.mean(target))
print(try Stats.variance(target, ddof: 1))
```

The mean is 5 and squared deviations sum to 20. Sample variance is 20/3 and population variance is 5. The general denominator is n minus the degrees-of-freedom adjustment, and it must be positive. Match estimator conventions before comparing higher moments. The release tests translate the SciPy reference to the SwiftSci sample-denominator convention before checking skewness and excess kurtosis (`skewness diff 2.39e-17`, `excess kurtosis diff 2.13e-14` on `100,000` points).

## Prepare data without leakage

Split observations before fitting imputers, scalers, encoders or dimensionality reduction. Fit on training rows and transform holdout rows using that state. Repeat the fit inside each cross-validation fold. Preserve feature names, order and units with saved models.

```swift
import SwiftPreprocessing

let training: [[Double]] = [[1, 10], [2, 15], [3, 20]]
let holdout: [[Double]] = [[4, 25]]
var scaler = StandardScaler()
try scaler.fit(training)
print(try scaler.transform(holdout))
```

This uses training coordinates for the holdout. Refitting on the holdout would change the coordinates and introduce leakage.

# Modeling, neural networks, and predictive evaluation

A typed table carries schema information through preparation. An estimator takes an ordered feature matrix and a target for each training row. Reusable prepared batches can reduce conversion when successive compatible operations use a common representation.

In `3.14.0`, `OneVsRestClassifier` conforms to `ClassifierEstimator` (`getBias`, `getWeights`, `getWeightsAndBias`, `decisionFunction`, and vectorized `predictProbability` at `0.069 ms` vs scikit-learn `0.203 ms`, **2.94×** faster), `RandomForestClassifier` adds `predictProbability(features:mode:)` (`ForestProbabilityMode.softLeafDistribution` and `.hardVote`), and `FlatTreeNode` records `cover` and `classDistribution` across trees and persisted `GradientBoostedTreesModelState`. These build on `3.13.3` Accelerate (`cblas_dgemm` / `vDSP`) mini-batch `MLPClassifier` / `MLPRegressor` training, multiclass `HistGradientBoostingClassifier` (`K > 2`), `LinearSVC` epoch/rectangularity guards, and `PlattScaling` / `CalibratedClassifier` binary stability.

```swift
import SwiftML

let features: [[Double]] = [[1], [2], [3], [4]]
let targets: [Double] = [2, 4, 6, 8]
let model = LinearRegression()
try await model.fit(features: features, targets: targets)
print(try await model.predict(features: [[5]]))
```

The noiseless example predicts approximately 10. For a real regression, report the split, model settings and target units. RMSE is the square root of the mean squared prediction error. For classification, inspect class balance and report a class-sensitive metric when aggregate accuracy hides errors. For clustering, numeric labels can be permuted without changing assignments, so judge stability or structure rather than raw IDs.

The [persisted numerical workflow](https://github.com/Nodibell/SwiftSci/tree/release/3.14.0/Examples/NumericalWorkflow) demonstrates training-only preprocessing, PCA, classification and prediction on unlabeled rows. Retain its feature schema with the fitted state.

# Forecasting and fitted values

A future forecast estimates unseen observations. An in-sample fitted value is a one-step conditional prediction at a historical observation. SwiftSci reconstructs ARIMA and SARIMA fitted values using observed levels. ARIMAX adds the exogenous contribution.

For zero-based index t, warmup is d for ARIMA and d + D times s for SARIMA. The fitted array and observed series have equal lengths. Warmup entries reproduce observations and have no innovations. At or after warmup, the identity holds within floating-point precision:

```text
residuals[t - warmUp] = series[t] - fittedValues[t]
```

```swift
import SwiftForecast

let series: [Double] = [8, 11, 10, 12, 14, 13, 15, 16, 19, 17, 20, 22]
let model = try ARIMAModel(p: 0, d: 1, q: 0)
try await model.fit(series: series)
let fitted = await model.fittedValues
let residuals = await model.residuals
let warmUp = await model.order.d
print(series[warmUp] - fitted[warmUp], residuals[0])
```

## Handle failure and evaluate forecasts

ARIMA/SARIMA reject non-finite fit intermediates and invalid moving-average polynomials and clear fitted state on failure. AutoARIMA applies the corrected coefficient sign convention during candidate validation. Align exogenous rows with training observations, and supply corresponding future values at forecast time. Construct each feature only from information available at the forecast origin.

Use increasing chronological cutoffs. Fit on each prefix, predict a fixed horizon and assess observations after the cutoff. Compare with a simple reference under the same protocol. In the recorded synthetic ARIMA backtest, seasonal naive achieved lower mean RMSE (4.276752) than either fitted implementation (5.223510 for SwiftSci and 5.071532 for Python). The one-step fitted-value checks establish an exact residual identity (`<= 5.68e-14`), while rolling-origin RMSE addresses predictive quality.

See the [forecasting guide](https://nodibell.github.io/SwiftSci/documentation/swiftforecast/arimaandsarima/) for the ARIMA and SARIMA interfaces.

# Explanations, vision, LLMs, and agents

Specify the model output, feature order and weighting convention before comparing explanations. SwiftSci's `TreeSHAP` enumerates coalitions exactly for trees with up to 12 distinct split features (with `2,048` seeded permutation samples for larger trees) and supports both `.uniformLeaf` and `.empiricalCover` (`TreeNodeWeighting`) node weights.

In `3.14.0`, **G-035** (`TreeSHAP` for `GradientBoostedTreesRegressor`) and **G-036** (per-class probability `TreeSHAP` via `explainClassDetailed`, `explainAllClassesDetailed`, and `explainBatchClass` on `DecisionTreeClassifier` and `RandomForestClassifier`) are implemented and verified against `2^K` coalition enumeration and Python `shap.TreeExplainer` (`shap 0.54.1`) across 5 models (`maxShapDiff <= 1.78e-15`).

In `SwiftVision`, `SwiftNLP`, `SwiftLLM`, and `SwiftAgent`, `3.14.0` adds `YOLODatasetMetadata` YAML parsing, image/background cropping (`ImageDataset.crop` / `cropAll` / `sampleBackgroundCrop`), `ImageDataset.loadFolder` / `loadSplitFolder` (**G-039**), the `TrainedVisionClassifier` actor with compiled Core ML `VisionFeaturePrint.Scene` + `MLModel` pipeline parity via `CoreMLExporter.exportVisionClassifierPipeline` (**G-040**), `ChatTemplate.defaultStopSequences`, `GenerationConfiguration` (`generation_config.json`), `LLMOptions.stopSequences` / `stopTokenIDs`, and chat-templated `ReActAgent.stream(query:chatTemplate:...)` (**G-038**).

# Read and reproduce the release measurements

The verified release used Apple M4 Pro (arm64, 24 GiB), macOS 27.2.1 build 26B131, Xcode 27.2 and Swift 6.4.2, on AC power. The standardized runner timed operations through materialized output. Fixture preparation, process startup and reference validation were outside measured samples. The values below are medians of independently measured process medians (`production-default`, AC power, 5 rotating processes per engine).

| Operation (`standard`, 100,000 rows) | SwiftSci | Pandas | Polars | DuckDB |
|---|---:|---:|---:|---:|
| Eager CSV read | 2.405 ms | 10.353 ms | 1.053 ms | 25.674 ms |
| Row filter | 0.230 ms | 0.731 ms | 0.667 ms | 0.990 ms |
| Flat feature matrix | 0.106 ms | 0.567 ms | 1.448 ms | 1.454 ms |
| Standard scaling | 2.033 ms | 0.842 ms | 1.504 ms | 2.409 ms |

In `v3140_bench` Release timings, `OneVsRestClassifier.predictProbability` (`500 × 16`, `4` classes) ran in `0.069 ms` median vs scikit-learn `0.203 ms` (**SwiftSci 2.94×** faster).

Compare workloads with equivalent outputs and thread policy. Scientific workloads report fit and inference separately. Very short operations can fall below timing resolution. Whole-process RSS includes libraries and warmup allocations. These figures do not imply a general speed or memory advantage.

## Check scope and provenance

1. Resolve the release branch or tag and keep its commit SHA and source-tree SHA-256.
2. Use the [runner guide](https://github.com/Nodibell/SwiftSci/blob/release/3.14.0/Benchmarks/README.md) to prepare a declared dataset and build the worker.
3. Match the profile, engine versions and thread settings. Keep the fixture and worker hashes.
4. Audit the run and record numerical tolerances before comparing timings.

Six AC-powered `production-default` standardized campaigns passed audit (`10,300` validated samples, `0` failures). Scientific validation passed `39/39` workloads and `3/3` rolling-origin backtests. SwiftSci passed all `202/202` CPU acceptance executions (`4` pandas Wampler 4/5 comparator failures retained their original tolerances with `0` regression blockers). Across `xcodebuild test` and `swift test -c release`, `1,919` total tests ran (`1,912` passed / `2,814` dynamic runs, `0` failures, `7` environment-gated skips), `15/15` DocC archives built with `--warnings-as-errors` (`2,641/2,641` declarations, `0` placeholders), and `CookbookVerification` passed `71/71` recipes.

[Performance](https://github.com/Nodibell/SwiftSci/blob/main/PERFORMANCE/3.14.0.md) · [Accuracy](https://github.com/Nodibell/SwiftSci/blob/main/ACCURACY/3.14.0.md) · [Reproducibility archive](https://github.com/Nodibell/SwiftSci/tree/main/Benchmarks/Reports/v3.14.0)
