# 🗺️ SwiftSci Architectural Roadmap (v1.0 – v3.1+)

## 📌 Vision & Architecture

**SwiftSci** is a high-performance modular ecosystem for scientific computing, data analysis, and machine learning natively optimized for **Apple Silicon (M-series, UMA)** and **Swift 6 Strict Concurrency**.

The architecture combines two hardware engines:
1. **CPU Engine**: SIMD vectorization via Apple **Accelerate (`vDSP`, `BLAS`, `LAPACK`)**.
2. **GPU Engine**: Unified Memory Architecture (UMA) tensor evaluation via **MLX Metal**.

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
7. **Client UI Integration**
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

### Version 2.3: SIMD Acceleration, FFT Spectral Engines & Enterprise Drivers *(🟢 Completed)*

*Detailed implementation plan:* [implementation_plan_23.md](implementation_plan_23.md)

1. **Phase 1**: SIMD Bitmask Boolean Filtering in `SwiftDataFrame` (`0.02×` → `>1.0× vs Pandas`).
2. **Phase 2**: Primitive Array `vDSP.sort` Radix Indexing in `SwiftDataFrame` (`0.10×` → `>1.0× vs Pandas`).
3. **Phase 3**: `vDSP` 1D FIR Convolution & Real FFT Spectral Engine in `SwiftForecast` (`0.09×` → `>1.0× vs Statsmodels`).
4. **Phase 4**: Real U-Net 2D Segmentation & YOLOv8 Detector in `SwiftVision`.
5. **Phase 5**: PostgreSQL Connection Driver in `SwiftDatabase`.
6. **Phase 6**: SIMD Tree Split Evaluation Vectorization with Accelerate `vDSP` in `SwiftML`.
7. **Phase 7**: Comprehensive Verification, Benchmark Suite & DocC Documentation Update.

---

### Version 2.3.1: Maintenance, Stability & SwiftNotebook Compatibility *(🟢 Completed)*

1. **Core Type Governance**: Fixed typealias ambiguity between `SwiftDataFrame.NLPError` and `SwiftNLP.NLPError`.
2. **Ecosystem Compatibility**: Full compatibility update for `SwiftNotebook` integration (`DataFrame+Plot.swift`).
3. **Memory Safety & Performance**: Optimized zero-copy column buffer passes and SIMD bitmask filter memory bounds.

---

### Version 2.4.0: Complete 14-Module DocC Documentation & NLTK-Equivalent SwiftNLP Engine *(🟢 Completed)*

*Detailed implementation plan:* [implementation_plan_24.md](implementation_plan_24.md)

1. **NLTK-Equivalent SwiftNLP Engine**:
   - **Tokenizers**: `AppleWordTokenizer` (Apple OS multi-lingual boundary detection), `SentenceTokenizer`, `RegexTokenizer`, `BPETokenizer`, `NGramTokenizer`.
   - **Linguistic Processing**: `PorterStemmer` (morphological suffix stripping), `POSTagger` (part-of-speech tagging), `AppleLemmaTagger` (canonical base form lemmatization).
   - **Entity Extraction**: `AppleNamedEntityRecognizer` (Person, Place, Organization).
   - **Sentiment Analysis**: Pure Swift `VADERSentimentAnalyzer` (backed by zero-latency pre-sorted `VADERLexicon` binary search), `NLSentimentAnalyzer` (Apple OS ML model).
   - **Language & Embeddings**: `AppleLanguageDetector`, `AppleNLEmbedding`, `WordEmbeddings` (SIMD Accelerate `vDSP_dotprD` dot product optimization).
   - **Text Classification**: `MultinomialNaiveBayes` & `ComplementNaiveBayes`.
2. **Ecosystem Inter-Module Integration Extensions**:
   - `TextPipeline` (SwiftML actor pipeline).
   - `df.vectorizeTextColumn` (SwiftCluster text clustering).
   - `LLMContextWindow` (SwiftLLM token counting & prompt truncation).
   - `TextExplainer` (SwiftExplain token importance scoring).
3. **Module-by-Module DocC Documentation (100% Coverage across 14 Targets)**:
   - Added comprehensive triple-slash (`///`) docstrings to all 14 targets (`SwiftDataFrame`, `SwiftStats`, `SwiftPreprocessing`, `SwiftML`, `SwiftCluster`, `SwiftNLP`, `SwiftOptimize`, `SwiftForecast`, `SwiftLLM`, `SwiftExplain`, `SwiftVisualization`, `SwiftVision`, `SwiftDatabase`, `SwiftAgent`).
   - Achieved **0 compiler documentation warnings** across all 14 workspace targets.
4. **Unified Static HTML DocC Website**:
   - Executed `./scripts/build_unified_docs.sh` compiling zero-warning static HTML documentation in `./docs/` with unified navigation index (`docs/index/index.json`).

---

### Version 2.5.0: Arrow IPC / Feather Serialization, LazyDataFrame, MultiOutput Models, ETSModel & KVCache *(🟢 Completed)*

*Detailed implementation plan:* [implementation_plan_25.md](implementation_plan_25.md)

1. **Feather / Arrow IPC Serialization (`SwiftDataFrame`)**:
   - High-performance binary file and buffer I/O (`FeatherReader`, `FeatherWriter`, `DataFrame.init(feather:)`, `writeFeather(to:)`).
2. **LazyDataFrame & Query Optimization (`SwiftDataFrame`)**:
   - Deferred execution pipeline (`DataFrame.lazy()`, `.filter()`, `.select()`, `.collect()`) with filter predicate merging and pushdown.
3. **MultiOutput Models & Generalized Hyperparameter Search (`SwiftML` & `SwiftOptimize`)**:
   - Parallel `MultiOutputRegressor` and `MultiLabelClassifier` actors.
   - Generalized `RandomizedSearchCV.searchGeneric` for parameter dictionary optimization across custom estimator factories.
4. **SwiftForecast Models (`SwiftForecast`)**:
   - `ETSModel` state space forecasting with R-equivalent `autoFit` AICc model selection.
   - Prophet-style `PiecewiseTrendDecomposition` (piecewise linear & logistic trends).
5. **SwiftLLM KV-Cache & Streaming (`SwiftLLM`)**:
   - `KVCache` Key-Value tensor cache for autoregressive inference.
   - `generateStream(prompt:options:)` returning `AsyncThrowingStream<String, any Error>`.
6. **Native Charting (`SwiftVisualization`)**:
   - `SwiftSciChartView` native SwiftUI `Canvas` charting component for line, bar, and heatmap visualization.
7. **`swiftsci` CLI Utility (`SwiftSciCLI`)**:
   - Command-line utility for dataset summaries (`swiftsci summary`), CSV/Feather format conversions (`swiftsci convert`), and model export inspection.
8. **Full Test Suite & Documentation Verification**:
   - 100% test pass rate across all 15 workspace targets and updated DocC documentation archives.

---

### Version 2.6.0 & 2.6.1: Swift 6 Strict Concurrency, Memory Safety & Benchmark Rigor *(🟢 Completed)*

1. **Swift 6 Concurrency & Sendability**:
   - Full concurrency audit across all 14 targets. Eliminated data races and unsafe pointer captures in `GradientBoosting`.
2. **Honest Benchmark Governance**:
   - Refactored benchmark suite preventing silent swallowing of benchmark failures or assertions.

---

### Version 2.6.2: Authentic 100.00% Public DocC API Coverage & Automated CI Enforcement *(🟢 Completed)*

*Detailed implementation plan:* [implementation_plan.md](implementation_plan.md)

1. **Authentic 100.00% DocC API Coverage**:
   - Added rich Markdown `///` docstrings with parameters, return types, exceptions, and runnable usage examples for all 1,176 public/open symbols across all 14 targets (`SwiftDataFrame`, `SwiftStats`, `SwiftPreprocessing`, `SwiftML`, `SwiftCluster`, `SwiftOptimize`, `SwiftForecast`, `SwiftNLP`, `SwiftExplain`, `SwiftLLM`, `SwiftVisualization`, `SwiftVision`, `SwiftDatabase`, `SwiftAgent`).
2. **Automated CI Coverage Verification**:
   - Created `scripts/verify_doc_coverage.py` parsing public/open Swift declarations and failing CI if coverage drops below 100.00%.
3. **Zero Compiler Documentation Warnings**:
   - Verified via `swift package generate-documentation --analyze`, producing 0 compiler documentation warnings across all targets.

---

### Version 2.7.0: Consolidation, Value Semantics & Feature Maturity *(🟢 Completed)*

*Detailed implementation plan:* [implementation_plan_27.md](implementation_plan_27.md)

1. **`SwiftPreprocessing` Value Semantics & Container Composition Fix**:
   - Refactored `MinMaxScaler`, `StandardScaler`, and `RobustScaler` from `final class: @unchecked Sendable` to **`struct: Sendable`**.
   - Updated `PreprocessingTransformer` protocol with `mutating func fit(_ data:) throws`, achieving strict Tier B value-semantics data-race freedom without `@unchecked Sendable`.
   - Fixed container mutation in `Pipeline`, `ColumnTransformer`, `ClassificationPipeline`, and `RegressionPipeline` to mutate elements directly by array index in `fit()`, ensuring fitted state persists in `steps` and `routes` for subsequent `transform()` / `predict()` calls on new data.
2. **`SwiftNLP` WordNet Synset & Semantic Similarity Engine**:
   - Added native WordNet synset lookup (`synsets(for:)`), hypernym/hyponym tree traversal (`hypernyms(of:)`, `hyponyms(of:)`), and shortest path / Wu-Palmer concept similarity metrics (`pathSimilarity`, `wupSimilarity`).
3. **`SwiftML` Binary ONNX Protobuf Exporter**:
   - Added `ONNXExporter.exportBinaryONNX` constructing binary ONNX `ModelProto` wire format bytes for cross-platform model deployment.
4. **`SwiftForecast` Error Handling & `SwiftDataFrame` Type Safety**:
   - Eliminated silent `catch { continue }` in `ExponentialSmoothing.swift` parameter optimization; throws `ForecastError.trainingFailed` on grid search failure.
   - Replaced all 11 instances of forced dynamic downcasting (`as!`) in `TypedColumn.swift` with safe `as?` conditional unwrapping.
5. **Concurrency Decision Matrix & CI Enforcement**:
   - Formalized Concurrency Tiers (A: `actor`, B: `struct`, C: `final class @unchecked Sendable`) and `Estimator` vs `PreprocessingTransformer` design guidelines in `CONTRIBUTING.md`.
   - Added `docc-check` gate (`swift package generate-documentation --warnings-as-errors`) to `.github/workflows/ci.yml`.
   - Configured `.spi.yml` for macOS-only builds.

---

### Version 2.8.0: Real YOLOv8 Object Detection Inference & ONNX Weight Parsing *(🟢 Completed)*

*Detailed implementation plan:* [implementation_plan_yolov8.md](implementation_plan_yolov8.md)

1. **Real YOLOv8n Neural Network Architecture (`SwiftVision`)**:
   - Implemented `YOLOBackbone` (CSPDarknet with `ConvBlock`, `BottleneckBlock`, `C2fBlock`, and `SPPFBlock` spatial pyramid pooling).
   - Implemented `YOLONeck` (PANet feature pyramid with top-down 2x upsampling and bottom-up strided convolutions).
   - Implemented `YOLOHead` (anchor-free decoupled classification and Distribution Focal Loss / DFL regression branches decoding 8,400 predictions).
   - Implemented `YOLOPreprocessor` (aspect-ratio letterbox resizing to 640x640 with `(114, 114, 114)` gray padding).
2. **ONNX Protobuf Binary Weight Reader (`SwiftVision` & `SwiftML`)**:
   - Implemented `ONNXWeightReader` parsing binary `.onnx` model graphs (`ModelProto` -> `GraphProto` -> `TensorProto` initializers) into `[String: MLXArray]`.
   - Integrated `YOLOWeightLoader` mapping PyTorch weight names (`model.0.conv.weight`, `model.0.bn.weight`...) directly into `SwiftVision` layer parameters.
3. **End-to-End Inference Engine (`SwiftVision`)**:
   - Replaced legacy heuristic image contrast placeholder in `YOLOv8Detector` with full, real GPU-accelerated forward pass execution via MLX / MLXNN on Apple Silicon.

---

### Version 3.0.0: API Contract Freeze: Error Unification, Duplication Removal & Concurrency Governance Completion *(🟢 Completed)*

*Detailed implementation plan:* [implementation_plan_30.md](implementation_plan_30.md)

1. **Error Type Consolidation**:
   - Consolidated `MLError`, `DataFrameError`, and `SwiftSciError` into `SwiftMLError`.
   - Deprecated `MLError` and `DataFrameError` typealiases with `@available(*, deprecated, renamed: "SwiftMLError")`.
   - Eliminated `SwiftSciError` across multi-output regressors, multi-label classifiers, estimator protocols, and AutoML.
2. **Shared Numeric Primitives (`Numerics.swift`)**:
   - Extracted clamped `sigmoid(_:)` with numerical overflow protection `min(50.0, max(-50.0, x))` and `Array.argmax()` extension into `Sources/SwiftML/Core/Numerics.swift`.
   - Replaced raw `exp` sigmoid & copy-pasted `argmax` one-liners across `LogisticRegression`, `LinearSVC`, `MLP`, `ImageDataset`, `CalibratedClassifier`, `OneVsRestClassifier`, and `LinearSVCOneVsRest`.
3. **Duplication Removal**:
   - Removed duplicate private `SeededRandom` from `OutlierDetection.swift` in favor of public `SwiftPreprocessing.SeededRandom`.
4. **Tier B Concurrency Migration (`SwiftPreprocessing`)**:
   - Migrated 12 `final class @unchecked Sendable` transformers (`VarianceThreshold`, `SelectKBest`, `RecursiveFeatureElimination`, `FrequencyEncoder`, `Imputer`, `KBinsDiscretizer`, `KNNImputer`, `MissingValueIndicator`, `Normalizer`, `PolynomialFeatures`, `PowerTransformer`, `TargetEncoder`) to `public struct ...: PreprocessingTransformer, Sendable`.
5. **`SwiftNLP` Classifier Protocol Governance**:
   - Introduced actor-based `NaiveBayesClassifier` and `ComplementNaiveBayesClassifier` conforming to `ClassifierEstimator` in `SwiftNLP`.
   - Deprecated struct originals `MultinomialNaiveBayes` and `ComplementNaiveBayes`.
6. **API Signature Cleanups & Protocol Conformance**:
   - Renamed and scoped `predictProbability1D` to internal `binaryPositiveClassProbability` in `LogisticRegression.swift`.
   - Added explicit `async` keyword to `DecisionTree` `fit`/`predict`/`predictProbability` signatures in `DecisionTreeClassifier` and `DecisionTreeRegressor`.
   - Closed `SystemsCSVParser` test gaps for escaped quotes and non-newline-terminated final lines.
   - Replaced manual mean reductions in `SwiftForecast` with `try Stats.mean(...)`.

---

### Version 3.0.1: Compatibility Patch — Dependency Pin, Driver Honesty & Agent DSL *(🟢 Completed)*

*Detailed implementation plan:* [implementation_plan_30_1.md](implementation_plan_30_1.md)

1. **FlatBuffers Resolution Pin (`SwiftDataFrame` dependency graph)**:
   - Pinned `FlatBuffers` to `exact: "25.2.10"` in `Package.swift` so downstream consumers resolve a build-safe graph with `arrow-swift` without a local override (G-004).
2. **`MySQLConnection` Stub & Driver Documentation Honesty (`SwiftDatabase`)**:
   - Added `MySQLConnection` symmetric to `PostgreSQLConnection`, throwing `DatabaseError.notImplemented` until native driver integration.
   - Updated `.docc` guides to distinguish implemented SQLite from PostgreSQL/MySQL stubs (G-003).
3. **SwiftAgent Cleaning Command Expansion (`SwiftAgent`)**:
   - Extended `AgentCommand` and `parseCommand` with `rename`, `dropnulls`, `fillnulls`, and `groupby` mapped to existing `SwiftDataFrame` APIs (`renameColumn`, row-null filtering via `gathered(at:)`, `TypedColumn.fillNull`, `GroupedDataFrame` aggregations) (G-002 partial).
4. **CoreMLExporter Documentation Alignment (`SwiftML`)**:
   - Corrected module doc to reflect JSON-only linear export; tree, forest, logistic, and binary `.mlmodel`/`.mlpackage` export remain deferred (G-001 partial).

**Deferred to future releases:**
- Native PostgreSQL (`libpq`) and MySQL wire-protocol drivers.
- Full SwiftAgent DSL (imputation, encoding, outlier treatment, joins, calculated columns).

---

### Version 3.1.0: Binary Core ML Export (G-001) *(🔵 Planned)*

*Detailed implementation plan:* [implementation_plan_coreml_binary.md](implementation_plan_coreml_binary.md)

1. **Shared `ProtobufWriter` (`SwiftML`)**:
   - Extract from `ONNXExporter.swift` for reuse across ONNX and Core ML wire-format encoders.
2. **Binary `.mlmodel` Export (`SwiftML`)**:
   - `GLMRegressor` / `GLMClassifier` for linear and logistic regression.
   - `TreeEnsembleClassifier` / `TreeEnsembleRegressor` for decision tree and random forest.
   - Validated via `MLModel(contentsOf:)` load and numerical parity with SwiftSci `predict`.
3. **Unified Export API (`SwiftML`)**:
   - `CoreMLExportable` conformance on fitted model types; deprecate JSON-only `exportLinearModel`.
4. **Gap closure**:
   - G-001 resolved for supported families; MLP/`NeuralNetwork` and `.mlpackage` deferred beyond 3.1.0.

---

## 🏛 Integration Guidelines for Client Applications

Thanks to its modular design, SwiftSci seamlessly integrates into applications following clean architecture:

* **View Models:** All model initialization, dataset loading (`SwiftDataFrame`), and preprocessing pipeline configurations reside in the View Model layer.
* **Background Tasks:** Method calls like `.fit()` for compute-heavy algorithms (e.g. Random Forest or `MLX` graph evaluations) should be executed inside isolated background tasks (`Task.detached { }`) to prevent main thread blocking and maintain smooth 120Hz UI rendering.
* **Complexity Encapsulation:** Low-level Arrow memory buffers and non-Sendable `MLXArray` handles are encapsulated as `internal`. Client applications interact strictly with thread-safe, public Swift 6 API contracts.


