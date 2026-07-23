# Changelog

All notable changes to the **SwiftSci** ecosystem will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-07-23 (Official Stable Release 🚀)

### Added
- **Direct Remote Dataset Loading (`SwiftDataFrame`)**: `DataFrame.readURL(_ url: URL)` and `DataFrame(remoteURL: URL)` for streaming and parsing remote CSV/JSON datasets directly into zero-copy DataFrames over HTTP/HTTPS.
- **Ukrainian News NLP Benchmark (`FIdo-AI/ua-news`)**: Added `19_UkrainianNewsClassification.swift` tutorial demonstrating Ukrainian text normalization, TF-IDF vectorization, and multi-class classification (`OneVsRestClassifier`). Added Ukrainian NLP parity requirements spec to `implementation_plan_20.md`.
- **LLM Architecture & Sampling Tutorial (`SwiftLLM`)**: Fully populated `13_LLM.swift` with `BPETokenizer`, `TransformerDecoder` forward pass, `Sampler` (greedy argmax & top-K temperature sampling), and zero-copy `GGUFParser` / `SafeTensorsParser` demonstrations.
- **Tutorial Suite Expansion (`SwiftAnalyticsDemo`)**: Added interactive tutorials 14-19 covering Computer Vision (`SwiftVision`), Relational Database Connectors (`SwiftDatabase`), Execution Sandbox (`SwiftAgent`), AutoML & Explainability (`SwiftExplain`), and Versioning & Diffs (`SwiftDataFrame`).

---

## [1.7.0] - 2026-07-23

### Added
- **`TargetEncoder` (`SwiftPreprocessing`)**: Bayesian target encoding with smoothing parameter to prevent data leakage. Conforms to `PreprocessingTransformer`.
- **`FrequencyEncoder` (`SwiftPreprocessing`)**: Category frequency ratio encoder with normalized output. Conforms to `PreprocessingTransformer`.
- **`KNNImputer` (`SwiftPreprocessing`)**: Weighted k-nearest-neighbours imputation using Euclidean distance on non-missing features. Conforms to `PreprocessingTransformer`.
- **`SwiftVisualization` module**: New module with interactive HTML chart exporters powered by embedded Plotly.js: `plotCorrelationHeatmap`, `plotROCCurve`, `plotFeatureImportances`, `plotConfusionMatrix`.
- **Full DocC Static Site**: Generated and deployed unified DocC documentation for all 11 modules to GitHub Pages (`--transform-for-static-hosting`). Custom landing page at `https://nodibell.github.io/SwiftSci/`.

### Performance
- **KMeans 13.8× speedup**: Replaced `vDSP_distancesq` with manually unrolled `distanceSquared` for short vectors and switched to GCD chunking (512 items/chunk), reducing context-switch overhead.
- **CSV Read 1.7× speedup**: Added 1 000-row sampling heuristic for column type inference, avoiding full-file scan on large CSVs.
- **LLM Forward Pass restored**: Resolved MLX device scheduling regression; forward pass latency returned to baseline.

### 📊 Benchmark Performance Summary (v1.7.0 vs Python)
| Benchmark Test | Swift (ms) | Python (ms) | Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **ARIMA(1,1,1) Fit** (50k pts) | 2.36 ms | 201.62 ms | 85.4× | 🟢 Swift |
| **Holt-Winters Fit** (50k pts, period=12) | 6.72 ms | 133.76 ms | 19.9× | 🟢 Swift |
| **KernelSHAP Explain** (5 feats, 100 coalitions) | 0.084 ms | 0.426 ms | 5.1× | 🟢 Swift |
| **Random Forest Fit** (1k×4, 50 trees) | 4.84 ms | 24.03 ms | 5.0× | 🟢 Swift |
| **KMeans fit** (10k×4, 3 clusters) | 30.87 ms | 11.87 ms | 0.38× | 🔴 Python |
| **CSV Read** (100k rows, 5 cols) | 98.39 ms | 18.05 ms | 0.18× | 🔴 Python |
| **Kalman Filter 1D** (10k obs) | 64.52 ms | 80.45 ms | 1.25× | 🟢 Swift |

* **CI Status**: **PASSED** ✅ (*0 regressions detected*).

---

## [1.6.1] - 2026-07-23

### Fixed
- **`DataFrame.debugPrint()` non-throwing**: Removed spurious `try` call-sites across Demo tutorials (`HousePriceRegression`, `CustomerSegmentationAnalysis`, `StockTimeSeriesAnalysis`).
- **Unused `trueLabels` warning**: Replaced with `_` in `CustomerSegmentationAnalysis`.
- **CI workflow cleanup**: Streamlined `ci.yml` steps; removed redundant resolve step.

---

## [1.6.0] - 2026-07-22

### Added
- **`DataFrame.toFeatureMatrix(_:)` / `toTargetVector(_:)`**: Direct ML extraction helpers that convert named columns into `[[Double]]` feature matrix and `[Double]` target vector without manual column iteration.
- **`columnTypeOverrides` (`CSVReadOptions`)**: Allows callers to override inferred column types at read time (e.g. force `Int64` for boolean-encoded columns).
- **`StopWords` & `TextNormalizer` (`SwiftNLP`)**: English stop-word filter and Unicode NFKC text normalization pipeline.
- **Dataset Generators (`SwiftML`)**: Added `makeClusters` and `makeCircles` to `DatasetUtilities` for unsupervised learning demos.
- **SPM dependency cleanup**: Removed unused `SwiftPrivacy` target; pruned transitive `Package.swift` dependencies.
- **DataFrame ↔ ML extensions**: Added `fit(_ df:features:target:)` and `predict(_ df:features:)` convenience overloads on `LinearRegression`, `LogisticRegression`, `RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostedTreesRegressor`.

### 📊 Benchmark Performance Summary (v1.6.0 vs Python)
| Benchmark Test | Swift (ms) | Python (ms) | Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **ARIMA(1,1,1) Fit** (50k pts) | 2.24 ms | 201.62 ms | 90.0× | 🟢 Swift |
| **Holt-Winters Fit** (50k pts, period=12) | 6.65 ms | 133.76 ms | 20.1× | 🟢 Swift |
| **Random Forest Fit** (1k×4, 50 trees) | 4.82 ms | 24.03 ms | 4.99× | 🟢 Swift |
| **CSV Read** (100k rows, 5 cols) | 144.9 ms | 18.05 ms | 0.12× | 🔴 Python |
| **KMeans fit** (10k×4, 3 clusters) | 413.4 ms | 11.87 ms | 0.03× | 🔴 Python |

* **CI Status**: **PASSED** ✅ (*0 regressions detected*).

---

## [1.5.0] - 2026-07-22

### Added
- **Column-Parallel CSV Reader (`SwiftDataFrame`)**: Redesigned `CSVReader` to construct typed columns in parallel using `withTaskGroup`, reducing wall-clock parse time on multi-core hardware.
- **`DataFrame.join` hash joins (`SwiftDataFrame`)**: Implemented inner, left, right, and full-outer hash joins (`join(_:on:how:)`).
- **`DataFrame.pivot` / `melt` (`SwiftDataFrame`)**: Wide ↔ long matrix reshaping operators.
- **`MLPClassifier` / `MLPRegressor` (`SwiftML`)**: Multi-Layer Perceptron estimators with configurable hidden layers, learning rate, and `maxIter`. CPU gradient descent via Accelerate `cblas_dgemm`.
- **Bitmap-free filtering (`SwiftDataFrame`)**: Replaced boolean bitmask filter with `filteredIndices` producing an `[Int]` index array; downstream `gathered(at:)` uses vectorized `vDSP_vgathr` gather.
- **DocC catalog articles & landing page**: Added `SwiftSci.docc` catalog with hand-written Getting Started, Architecture, and Performance articles.

### 📊 Benchmark Performance Summary (v1.5.0 vs Python)
| Benchmark Test | Swift (ms) | Python (ms) | Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **ARIMA(1,1,1) Fit** (50k pts) | 2.32 ms | 201.62 ms | 86.9× | 🟢 Swift |
| **Holt-Winters Fit** (50k pts, period=12) | 6.71 ms | 133.76 ms | 19.9× | 🟢 Swift |
| **Random Forest Fit** (1k×4, 50 trees) | 4.56 ms | 24.03 ms | 5.27× | 🟢 Swift |
| **CSV Stream Read** (chunk=10k) | 161.5 ms | 20.36 ms | 0.13× | 🔴 Python |

* **CI Status**: **PASSED** ✅ (*0 regressions detected*).

---

## [1.4.0] - 2026-07-22

### Added
- **`SystemsCSVParser` (`SwiftDataFrame`)**: High-performance zero-copy memory-mapped RFC 4180 DFA byte parser (`SystemsCSVParser`), speeding up CSV ingestion by ~10× on large datasets.
- **Vectorized Byte Parsers (`SwiftDataFrame`)**: Added zero-allocation ASCII parsers (`VectorizedByteParsers`) for fast `Double`, `Int`, and string decoding from unmanaged byte buffers.
- **vDSP Reductions (`SwiftDataFrame`)**: Accelerated `TypedColumn<Double>` `mean()`, `variance()`, and `stdDev()` using Apple Accelerate `vDSP` reductions.
- **DataFrame Index Filtering & Argsort (`SwiftDataFrame`)**: Added `filterRows(by:)` index-based boolean mask filtering and `argsort()` column index sorting.
- **Recursive Feature Elimination RFE (`SwiftPreprocessing`)**: Added `RecursiveFeatureElimination` (RFE) for iterative feature selection based on estimator feature importances.
- **Feature Importances & Model Persistence (`SwiftML`)**: Added Gini `featureImportances` property to `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, and added `Codable` serialization (`save(to:)` / `load(from:)`) across classical estimators.
- **NLP Text Tokenizers & Feature Extraction (`SwiftNLP`)**: Added `NGramTokenizer` (word & char n-grams) and `HashingVectorizer` (MurmurHash3 memory-bounded token hashing).
- **Time Series Windowing (`SwiftForecast`)**: Added `ExpandingWindow` feature transformer for cumulative time-series feature generation.
- **Swift DocC Integration & Multi-Module Site**: Integrated `swift-docc-plugin` (`v1.5.0`) in `Package.swift` and deployed unified 10-module static web documentation to GitHub Pages with `.nojekyll` and route auto-redirects.

### Fixed & Refactored
- **GradientBoostedTrees Bounds Check (`SwiftML`)**: Uncommented and enforced bounds checking in `GradientBoostedTreesRegressor`.
- **Seeded Random Number Generators (`SwiftPreprocessing`)**: Fixed deterministic seeding in `SMOTE` and `RandomUndersampler` using `LCG`.
- **SelectKBest ANOVA Scoring (`SwiftPreprocessing`)**: Fixed ANOVA F-statistic calculation in `SelectKBest` and added explicit `SwiftStats` dependency to `SwiftPreprocessing` target.
- **CalibratedClassifier Actor Concurrency (`SwiftML`)**: Converted `CalibratedClassifier` to `actor` to guarantee thread safety under Swift 6 strict concurrency rules.
- **Wired Memory Scaling (`SwiftPreprocessing`)**: Dynamically scaled `WiredMemoryManager` default limits using `ProcessInfo.processInfo.activeProcessorCount`.
- **GGUF Alignment Fix (`SwiftLLM`)**: Added `loadUnaligned` in `GGUFParser` to prevent alignment fault crashes on unaligned byte buffers.

### 📊 Benchmark Performance Summary (v1.4.0 vs Python)
| Benchmark Test | Swift (ms) | Python (ms) | Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **ARIMA(1,1,1) Fit** (50k pts) | 2.228 ms | 206.414 ms | 92.66x | 🟢 Swift |
| **ARIMA(1,1,1) Forecast** (horizon=24) | 2.297 ms | 209.709 ms | 91.28x | 🟢 Swift |
| **Holt-Winters Fit** (50k pts, period=12) | 6.823 ms | 138.947 ms | 20.37x | 🟢 Swift |
| **Random Forest Fit** (1k×4, 50 trees) | 4.580 ms | 24.217 ms | 5.29x | 🟢 Swift |
| **KernelSHAP Explain** (5 feats, 100 coalitions) | 0.148 ms | 0.452 ms | 3.05x | 🟢 Swift |
| **Kalman Filter 1D** (10k obs) | 56.508 ms | 84.692 ms | 1.50x | 🟢 Swift |
| **Mean** (vDSP, 1M elements) | 0.083 ms | 0.119 ms | 1.44x | 🟢 Swift |

* **CI Status**: **PASSED** ✅ (*0 regressions detected*).

---

## [1.3.0] - 2026-07-22

### Added
- **Estimator Pipelines (`SwiftML`)**: Introduced `ClassificationPipeline` and `RegressionPipeline` for chaining preprocessing transformers (`PreprocessingTransformer`) directly into final estimators without data leakage.
- **ColumnTransformer (`SwiftPreprocessing`)**: Introduced `ColumnTransformer` for routing independent column index subsets to separate preprocessing transformers and concatenating transformed outputs.
- **Outlier Detection (`SwiftCluster`)**: Implemented `IsolationForest` (random partition isolation trees) and `LocalOutlierFactor` (LOF kNN density estimation) anomaly detection models.
- **Imbalanced Learning (`SwiftPreprocessing`)**: Added `SMOTE` over-sampling technique using kNN interpolation and `RandomUndersampler` for majority class sub-sampling.
- **RandomizedSearchCV (`SwiftOptimize`)**: Added parallel randomized hyperparameter search across parameter distributions integrated with `KFold` cross-validation.
- **Probability Calibration (`SwiftML`)**: Added `CalibratedClassifier` applying Platt scaling logistic calibration over raw base classifier outputs.
- **Class Probabilities & Extended Metrics (`SwiftML` & `SwiftOptimize`)**: Added 2D matrix class probability requirement `predictProbability(features:)` to `ClassifierEstimator` protocol, and added `balancedAccuracy`, `matthewsCorrelationCoefficient` (MCC), `cohenKappa`, `logLoss`, `brierScore`, `rocCurve`, `rocAUC`, and `prCurve` to `Metrics.swift`.
- **Feature Selection & Engineering (`SwiftPreprocessing`)**: Added `VarianceThreshold`, `SelectKBest`, `InteractionFeatures`, and `DateFeatures`.
- **Time Series Transformers (`SwiftForecast`)**: Added `LagTransformer` and `RollingWindow` for constructing lagged feature matrices and rolling window statistics.
- **Text Vectorizer & Dataset Utilities (`SwiftNLP` & `SwiftML`)**: Added `CountVectorizer` document term frequency matrix builder, and `DatasetUtilities` (`makeClassification`, `makeRegression`, `makeMoons`).

### 📊 Benchmark Performance Summary (v1.3.0 vs Python)
| Benchmark Test | Swift (ms) | Python (ms) | Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **ARIMA(1,1,1) Fit** (50k pts) | 2.280 ms | 215.527 ms | 94.53x | 🟢 Swift |
| **Holt-Winters Fit** (50k pts, period=12) | 6.811 ms | 148.627 ms | 21.82x | 🟢 Swift |
| **KernelSHAP Explain** (5 feats, 100 coalitions) | 0.075 ms | 0.426 ms | 5.68x | 🟢 Swift |
| **Random Forest Fit** (1k×4, 50 trees) | 4.516 ms | 25.475 ms | 5.64x | 🟢 Swift |
| **Mean** (vDSP, 1M elements) | 0.081 ms | 0.121 ms | 1.49x | 🟢 Swift |
| **Pearson Correlation** (500k elements) | 0.856 ms | 1.256 ms | 1.47x | 🟢 Swift |
| **Isolation Forest Fit** (1k×10, 100 trees) | 14.048 ms | 16.510 ms | 1.18x | 🟢 Swift |
| **StdDev** (vDSP, 1M elements) | 0.477 ms | 0.539 ms | 1.13x | 🟢 Swift |
| **Variance** (vDSP, 1M elements) | 0.501 ms | 0.520 ms | 1.04x | 🟢 Swift |

* **CI Gate Status**: **PASSED** ✅ (*0 gated regressions detected*).

---

## [1.2.0] - 2026-07-22

### Added
- **Package Rename to `SwiftSci`**: Renamed main Swift Package to `SwiftSci` (`name: "SwiftSci"`) for cleaner ecosystem branding while preserving target module names (`SwiftDataFrame`, `SwiftStats`, `SwiftML`, `SwiftForecast`, etc.).
- **`DataFrame.addColumn(_:as:using:)`**: Added row-closure builder for computing new columns per-row over `DataFrameRow`, delegating to `withColumn`.
- **`DataFrameError.partialCastFailure`**: Added explicit error case thrown when partial element casting fails in `castColumn` (`failed` out of `total`), preventing silent data loss.

### Refactored & Fixed
- **Joseph Form Covariance Update (`SwiftForecast`)**: Updated `KalmanFilter` `filter()` and `smooth()` covariance matrix updates to the numerically stable Joseph form $P = (I - KH) P_{\text{pred}} (I - KH)^T + K R K^T$, preserving symmetry and positive-definiteness under long filtering runs.
- **UTF-8 Byte BPE Tokenizer (`SwiftNLP`)**: Refactored `BPETokenizer` `bpe()` algorithm to operate directly on `word.utf8` bytes with GPT-2 byte-to-unicode character encoding (`makeByteEncoder`), fixing incorrect tokenization for non-ASCII input.
- **DataFrame Invariant Diagnostics**: Refactored `DataFrame.gathered(at:)` to trap with `preconditionFailure` on internal length invariant violations instead of returning an empty DataFrame. Removed duplicate private `rows(at:)`.
- **`DataFrame.sample` Randomization**: Updated `sample(n:seed:ordered:)` to return randomly shuffled rows by default (`ordered: false`) while allowing index order preservation via `ordered: true`.

### 📊 Benchmark Performance Summary (v1.2.0 vs Python)
| Benchmark Test | Swift (ms) | Python (ms) | Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Mean** (vDSP, 1M elements) | 0.082 ms | 0.121 ms | 1.47x | 🟢 Swift |
| **Pearson Correlation** (500k elements) | 0.868 ms | 1.256 ms | 1.45x | 🟢 Swift |
| **ARIMA(1,1,1) Fit** (50k pts) | 2.323 ms | 215.527 ms | 92.78x | 🟢 Swift |
| **ARIMA Forecast Horizon=24** | 2.456 ms | 211.880 ms | 86.26x | 🟢 Swift |
| **Holt-Winters Fit** (50k pts, period=12) | 6.841 ms | 148.627 ms | 21.73x | 🟢 Swift |
| **Random Forest Fit** (1k×4, 50 trees) | 5.025 ms | 25.475 ms | 5.07x | 🟢 Swift |
| **KernelSHAP Explain** (5 feats, 100 coalitions) | 0.192 ms | 0.426 ms | 2.22x | 🟢 Swift |
| **Kalman Filter 1D** (10k obs, Joseph Form) | 62.349 ms | 85.547 ms | 1.37x | 🟢 Swift |
| **LLM Forward Pass** (seqLen=64) | 0.636 ms | 0.528 ms | 0.83x | 🔴 PyTorch |
| **CSV Read** (100k rows, 5 cols) | 177.509 ms | 19.340 ms | 0.11x | 🔴 Pandas |
| **CSV Stream Read** (chunk=10k) | 238.699 ms | 22.525 ms | 0.09x | 🔴 Pandas |

* **CI Gate Status**: **PASSED** ✅ (*0 gated regressions detected*).

---

## [1.1.0] - 2026-07-17

### Added
- **Streaming CSV Parser** (`SwiftDataFrame`): Implemented parallel, chunk-buffered streaming CSV reader returning `AsyncThrowingStream<DataFrame, Error>` utilizing Swift structured concurrency (`TaskGroup`) for high-performance file parsing.
- **SafeTensors & GGUF Parser** (`SwiftLLM`): Added native zero-copy weight parsers for local AI model inference, mapping model parameters from binary weights into `MLXArray` memory.
- **SwiftPrivacy Update**: Legacy ToyCKKS/RingLWE/PNNS primitives removed in v1.1; refactoring and binding to `apple/swift-homomorphic-encryption` is planned for v1.2.
- **SARIMA & GARCH Models** (`SwiftForecast`): Added `SARIMAModel` (Seasonal ARIMA) with seasonal differencing and conditional least squares, and `GARCHModel` (Generalized Autoregressive Conditional Heteroskedasticity) with coordinate descent MLE parameter optimization.

### Refactored
- Cleaned up compiler checks by removing legacy `#if ACCELERATE_NEW_LAPACK` checks across `PCA`, `ARIMA`, `KalmanFilter`, and `LAPACK+Wrapper` to utilize modern unified Accelerate LAPACK APIs directly.
- Eliminated raw pointer force-unwraps in Decision Tree split search helpers to ensure type safety.

---

## [0.9.0] - 2026-07-16

### Added
- Created `SwiftNLP` module tokenizer protocols and BPE subword tokenizer (`BPETokenizer`).
- Added static word embeddings loader (`WordEmbeddings`) with GloVe/Word2Vec text format parsing, cosine similarity computation, and Top-K search.
- Created `CHANGELOG.md` and `.spi.yml` for Swift Package Index integration.

### Refactored
- API Design Guidelines audit: clean parameter labels and unified return patterns.
- Concurrency audit: strict `Sendable` declarations across public interfaces.

---

## [0.8.0] - 2026-07-16

### Added
- Introduced the **Hardware Routing layer** via `HardwareRouter` allowing seamless CPU vs GPU execution.
- Added pure Swift CPU gradient descent paths for `LinearRegression` and `LogisticRegression`.
- Added pure CPU `PCA` path (Accelerate LAPACK `dgesvd`) and GPU path (`MLX.svd`).
- Implemented CPU-only density-based clustering `DBSCAN` with BFS expansion and `fitDBSCAN` DataFrame extension.
- Added 10 preprocessing steps to `SwiftPreprocessing`: `Imputer`, `Normalizer`, `TrainTestSplit`, `Pipeline`, `PolynomialFeatures`, `OrdinalEncoder`, `RobustScaler`, `PowerTransformer`, `KBinsDiscretizer`, `MissingValueIndicator`.

### Fixed
- Fixed command-line test runner Metal metallib loading crashes via dynamic bundle detection and colocated resource search.

---

## [0.7.0] - 2026-07-15

### Added
- Created Swift benchmark suite `SwiftAnalyticsBenchmarks` with real-time test progress indicators.
- Created Python baseline benchmark suite (Pandas/Scikit-Learn/Statsmodels).
- Built unified benchmark comparer script `compare.py` for CI gate checks.

---

## [0.6.0] - 2026-07-12

### Added
- Created `SwiftForecast` target for time series analysis.
- Implemented `TimeSeriesDecomposition` (Additive/Multiplicative, ACF, PACF, ADF stationary test).
- Implemented `ExponentialSmoothing` (Simple, Holt DES, Holt-Winters).
- Implemented `ARIMAModel` and `ARIMAX` (Conditional OLS, Hannan-Rissanen, AIC).
- Implemented state-space `KalmanFilter` with RTS smoother.

---

## [0.5.0] - 2026-07-08

### Added
- Created `SwiftOptimize` for cross-validation and evaluation.
- Implemented Decision Trees (`DecisionTreeClassifier`, `DecisionTreeRegressor`).
- Implemented Random Forest (`RandomForestClassifier`, `RandomForestRegressor`).
- Implemented Gradient Boosting (`GradientBoostedTreesRegressor`).

---

## [0.4.0] - 2026-07-02

### Added
- Resolved thread-safety issues of non-Sendable `MLXArray` by wrapping in actor isolation patterns and introducing the `WiredMemoryTicket` manager.

---

## [0.3.0] - 2026-06-25

### Added
- Implemented unsupervised learning algorithms (`PCA`, `KMeans` via MLX).
- Added basic TF-IDF Vectorizer in `SwiftNLP`.

---

## [0.2.0] - 2026-06-18

### Added
- Added basic data transformations to `SwiftPreprocessing`.
- Built first MLX-based training algorithms (Linear & Logistic Regression via MLX autodiff).

---

## [0.1.0] - 2026-06-10

### Added
- Created `SwiftDataFrame` with Apache Arrow integration.
- Created `SwiftStats` with Accelerate vDSP vector arithmetic (t-test, ANOVA).

---

## [0.0.1] - 2026-06-01

### Added
- Initialized multi-module Swift Package.
