# Changelog

All notable changes to the **SwiftSci** ecosystem will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
