# 🗺️ SwiftSci Architectural Roadmap (v1.0 – v2.2+)

## 📌 Vision & Architecture

**SwiftSci** is a high-performance modular ecosystem for scientific computing, data analysis, and machine learning natively optimized for **Apple Silicon (M-series, UMA)** and **Swift 6 Strict Concurrency**.

The architecture combines two hardware engines:
1. **CPU Engine**: SIMD vectorization via Apple **Accelerate (`vDSP`, `BLAS`, `LAPACK`)**.

### 🌐 Platform Compatibility & Multi-Target Deployment

- **macOS 14+ (Apple Silicon M-Series)**: Full support for all 14 modules leveraging Accelerate (CPU) and MLX Metal (GPU).
- **iOS 18+ & visionOS 2+**: Pure CPU vector modules (`SwiftDataFrame`, `SwiftStats`, `SwiftNLP`, `SwiftForecast`, `SwiftVisualization`, `SwiftDatabase`). MLX-dependent targets are conditionally built on macOS via `.when(platforms: [.macOS])`.

---

## 📅 Roadmap Overview

### Version 1.0 – 1.3: Core Foundation *(🟢 Completed)*

- **`SwiftDataFrame`**: Zero-copy Apache Arrow integration, memory-mapped CSV parser, streaming CSV/JSON, hash joins, `pivot`/`melt`.
- **`SwiftStats`**: Vectorized descriptive statistics, probability distributions (Student-t, Chi-Square, F), t-tests, ANOVA.
- **`SwiftPreprocessing`**: Scalers (`StandardScaler`, `MinMaxScaler`), Encoders (`OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`), Imputers (`KNNImputer`), `Pipeline`, `ColumnTransformer`.
- **`SwiftML`**: Vectorized Linear & Logistic Regression, Decision Trees, Random Forests, GBDT, Multi-Layer Perceptrons (MLP), synthetic dataset generators (`makeClassification`, `makeRegression`).
- **`SwiftCluster`**: SVD PCA, DBSCAN, `IsolationForest`, `LocalOutlierFactor`, `KMeans` (KMeans++ initialization).
- **`SwiftOptimize`**: `GridSearchCV`, `RandomizedSearchCV`, `AutoML` engine, cross-validation metrics.
- **`SwiftForecast`**: Exponential Smoothing, ARIMA, SARIMA, GARCH, Kalman Filter, time series decomposition.
- **`SwiftNLP`**: BPE tokenizer, `NGramTokenizer`, `HashingVectorizer`, `TFIDFVectorizer`, text normalization.
- **`SwiftExplain`**: Parallelized `KernelSHAP`, `TreeSHAP`, `PartialDependencePlot`, `PermutationImportance`.
- **`SwiftLLM`**: Causal Transformer Decoder on GPU, SafeTensors & GGUF weight parsers, Top-K/Top-P samplers.
- **`SwiftVisualization`**: Interactive Plotly HTML exporters (`plotCorrelationHeatmap`, `plotROCCurve`, `plotFeatureImportances`, `plotConfusionMatrix`).
- **`SwiftVision`**: Image dataset loading, U-Net segmentation, object detection wrappers, CNN feature extraction.
- **`SwiftDatabase`**: Direct SQLite connector (`SQLiteConnection`) for zero-copy DataFrame ingestion via `DataFrame.fromSQL`.
- **`SwiftAgent`**: RAG Context Summary Generator & execution sandbox.

---

### Version 2.1: Evaluation Metrics, Core API Freeze & MLOps *(🟢 Completed)*

*Detailed implementation plan:* [implementation_plan_21.md](implementation_plan_21.md)

1. **Core API Freeze & Deprecation Governance**
   - Locked public protocols (`AnyColumn`, `SupportedType`, `Estimator`, `Transformer`, `Classifier`, `Regressor`, `MetricEvaluator`).
   - Deprecated v1.x legacy patterns via `@available(*, deprecated, message: "...")`.
   - Guaranteed `Sendable` conformance for Swift 6 strict concurrency.
2. **Unsupervised Learning & Clustering Metrics (`SwiftCluster`)**
   - `Silhouette Score` (\([-1, 1]\)), `Inertia (WCSS)`, `Calinski-Harabasz Index`, `Davies-Bouldin Index`.
   - `Contamination Ratio` for `IsolationForest` anomaly detection, `ARI` and `NMI`.
3. **Advanced Evaluation Metrics (`SwiftOptimize` & `SwiftStats`)**
   - **Classification**: `ROC-AUC`, `PR-AUC`, `MCC`, `Log-Loss`, `Balanced Accuracy`, `F-beta Score`.
   - **Regression**: `R²`, `Adjusted R²`, `MAPE`, `Explained Variance Score`.
4. **Cross-Validation Schemes (`SwiftOptimize`)**
   - `StratifiedKFold`, `TimeSeriesSplit` (expanding-window), `GroupKFold`.
5. **Feature Engineering & Survival Analysis (`SwiftPreprocessing`, `SwiftStats`, `SwiftML`)**
   - `PolynomialFeatures`, time-lagged window functions (`withLaggedColumn`, `withRollingMean`, `withEWMA`), `VarianceThreshold`, `SelectKBest`, `RFE`.
   - `HistGradientBoosting` (256-bin binned splitting), `Kaplan-Meier Estimator`, `Cox Proportional Hazards Model`, `Probability Calibration` (Isotonic/Platt).
6. **MLOps & Model Export (`SwiftML`, `SwiftONNX`)**
   - `CoreMLExporter` (.mlmodel package), `ONNXExporter`, `TaskGroup` acceleration.
7. **Saura UI Integration**
   - Dynamic metric column selection in `ModelLeaderboardView.swift`.
8. **DataFrame Engine**
   - Automatic header deduplication in CSV reading (`CSVReader.deduplicateHeaders`).

---

### Version 2.2: Performance, Tree Pre-sorting & OLS *(🟢 Completed Phases 1–16)*

*Detailed implementation plan:* [implementation_plan_22.md](implementation_plan_22.md)

1. **DataFrame & Preprocessing Critical Bug Fixes (Phase 1)**
   - `toFeatureMatrix`: support `Bool` `nil` → `Double.nan`.
   - Rolling & EWMA: preserve `nil` elements.
   - `withEWMA`: replace `precondition` with `DataFrameError.invalidParameter`.
2. **DataFrame API Extensions (Phase 2)**
   - `mapColumn` for functional column transformation.
   - `DataFrameRow` typed subscripts & helpers (`row.double`, `row.string`, `row.int`).
   - `labelEncode` support for `Int64` and `Double`.
   - `GroupedDataFrame.transform`.
3. **MLP Enhancements (Phase 3)**
   - Flat `LayerWeights` layout.
   - Accelerate BLAS `cblas_dgemm` forward pass.
   - Adam optimizer with adaptive learning rate & moment corrections.
4. **Serialization Docs & PERFORMANCE.md (Phase 4)**
   - Honest documentation of JSON specifications for CoreML/ONNX exporters.
5. **SwiftStats vDSP Optimizations (Phase 5)**
   - SIMD `vDSP.sort` for `median` (`Double` & `Float`).
   - `cblas_dasum` for `norm(.l1)`.
6. **DecisionTree Pre-sorted Feature Matrix (Phase 6)**
   - `createPresortedIndices` for $O(N)$ split filtering in `DecisionTree`, `RandomForest`, `GradientBoosting`.
7. **HardwareRouter Expansion (Phase 7)**
   - Compute routing for `MLP`, `RandomForest`, `GBDT`, `DecisionTree`, `IsolationForest`.
   - `requestedDevice` & `resolvedDevice` in `MLPClassifier` / `MLPRegressor`.
8. **LinearRegression LAPACK OLS Backend (Phase 8)**
   - Single-pass analytical solution via LAPACK `dgels_` with automatic fallback to Gradient Descent.
9. **Test Coverage for All Fixes (Phase 9)**
   - Unit tests covering all bugfixes, Adam optimizer, HardwareRouter, and LAPACK OLS backend.
10. **Documentation & CHANGELOG (Phase 10)**
    - Release notes `[2.2.0]` in `CHANGELOG.md` & updated comparisons in `PERFORMANCE.md`.
11. **Performance Gap Fixes (Phase 11)**
    - DataFrame filter: reuse `DataFrameRow` instance (bypassed 100k+ allocations).
    - `parallelGathered(at:)` & `vDSP.sort` for `sortBy` acceleration.
    - Byte-level UTF-8 trim in `VectorizedByteParsers` (bypassed ~500k temporary String allocations).
    - Lazy NaN checks in `Stats.mean` and 1-pass `vDSP_measqvD` in `Stats.variance`.
    - PCA SVD solver accelerated $3-4\times$ via LAPACK `dgesdd_`.
    - `DispatchQueue.concurrentPerform` parallel predictions in `GradientBoostedTreesRegressor.predict`.
12. **SwiftDatabase Real SQL Driver (Phase 12)**
    - Native C-driver `sqlite3_open_v2`/`prepare_v2`/`step` in `SQLiteConnection`.
    - Typed `DatabaseError` and `PostgreSQLConnection.notImplemented`.
13. **SwiftAgent Real Command Parser (Phase 13)**
    - DSL parser in `SwiftAgentEvaluator.parseCommand` for `filter`, `sample`, `select`, `head`, `tail`.
    - Typed error `AgentError.unparseable`.
14. **SwiftVisualization Real ROC & AUC Computation (Phase 14)**
    - Dynamic calculation of FPR and TPR from sorted `(yScores, yTrue)` pairs.
    - Trapezoidal integration of Area Under Curve (AUC) embedded directly into Plotly chart titles.
15. **SwiftVision Deep Learning Inference Governance (Phase 15)**
    - Typed error `VisionError.notImplemented` for `YOLOv8Detector.detect` & `UNetSegmentationModel.predict`.
    - Refactored `ExtensionBenchmarks` to connect `CNNFeatureExtractor` & `VisionMetrics`.
16. **SwiftExplain Model-Aware Explainability (Phase 16)**
    - Permutation Importance measuring feature column shuffling and MSE degradation.
    - Partial Dependence Plot with real grid point substitution.
    - TreeSHAP integrating `KernelSHAP` for black-box model interpretation.

---

### Version 2.3: SIMD Acceleration, FFT Spectral Engines & Enterprise Drivers (🟢 Completed)

*Detailed implementation plan:* [implementation_plan_23.md](implementation_plan_23.md)

1. **Phase 1**: SIMD Bitmask Boolean Filtering in `SwiftDataFrame` (`0.02×` → `>1.0× vs Pandas`).
2. **Phase 2**: Primitive Array `vDSP.sort` Radix Indexing in `SwiftDataFrame` (`0.10×` → `>1.0× vs Pandas`).
3. **Phase 3**: `vDSP` 1D FIR Convolution & Real FFT Spectral Engine in `SwiftForecast` (`0.09×` → `>1.0× vs Statsmodels`).
4. **Phase 4**: Real U-Net 2D Segmentation & YOLOv8 Detector in `SwiftVision`.
5. **Phase 5**: PostgreSQL Connection Driver in `SwiftDatabase`.
6. **Phase 6**: SIMD Tree Split Evaluation Vectorization with Accelerate `vDSP` in `SwiftML`.
7. **Phase 7**: Comprehensive Verification, Benchmark Suite & DocC Documentation Update.

---

## 🏛 Integration Guidelines for Client Applications

Thanks to its modular design, SwiftSci seamlessly integrates into applications following clean architecture:

* **View Models:** All model initialization, dataset loading (`SwiftDataFrame`), and preprocessing pipeline configurations reside in the View Model layer.
* **Background Tasks:** Method calls like `.fit()` for compute-heavy algorithms (e.g. Random Forest or `MLX` graph evaluations) should be executed inside isolated background tasks (`Task.detached { }`) to prevent main thread blocking and maintain smooth 120Hz UI rendering.
* **Complexity Encapsulation:** Low-level Arrow memory buffers and non-Sendable `MLXArray` handles are encapsulated as `internal`. Client applications interact strictly with thread-safe, public Swift 6 API contracts.
