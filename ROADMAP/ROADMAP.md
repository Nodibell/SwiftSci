# 🗺️ SwiftSci Architectural Roadmap (v1.0 – v3.6+)

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

### Version 3.1.0: Binary Core ML Export (G-001 Full Resolution) *(🟢 Completed)*

*Detailed implementation plan:* [implementation_plan_coreml_binary.md](implementation_plan_coreml_binary.md)

1. **Shared `ProtobufWriter` (`SwiftML`)**:
   - Extracted zero-dependency protobuf encoder to `ProtobufWriter.swift`, shared across ONNX and Core ML serializers.
2. **Binary `.mlmodel` Export (`SwiftML`)**:
   - `GLMRegressor` for linear regression and `GLMClassifier` for binary logistic regression with logit post-evaluation transform.
   - `TreeEnsembleClassifier` for decision tree and random forest classifiers (with multi-tree encoding).
   - `TreeEnsembleRegressor` for decision tree and random forest regressors.
3. **Unified Export API (`SwiftML`)**:
   - Added `CoreMLExportable` protocol with asynchronous `exportCoreML(featureNames:outputName:)` and `writeCoreML(to:featureNames:outputName:)`.
   - Conformed `LinearRegression`, `LogisticRegression`, `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, and `RandomForestRegressor`.
   - Deprecated JSON-only `exportLinearModel`.
4. **Gap closure**:
   - G-001 resolved for supported model families; MLP/`NeuralNetwork` and `.mlpackage` directory bundles explicitly deferred beyond 3.1.0.

---

### Version 3.2.0: Full Feature Completeness, Native DB Protocols & Deep Vision *(🟢 Completed)*

1. **Native PostgreSQL & MySQL Wire Protocol Drivers (`SwiftDatabase`)**:
   - Implemented pure-Swift network wire-protocol drivers (`PostgresWireClient` v3.0 protocol and `MySQLWireClient` Client/Server protocol) with zero external C dependencies, fully eliminating `notImplemented` stubs.
2. **Deep Convolutional U-Net Architecture (`SwiftVision`)**:
   - Built a real deep U-Net semantic segmentation convolutional network (`UNetArchitecture`, `UNetDoubleConv`, `UNetDown`, `UNetUp`, `UNetOutConv`) on Apple Silicon MLX GPU/CPU.
3. **Binary Core ML NeuralNetwork Export (`SwiftML`)**:
   - Implemented native binary Protobuf encoding for Core ML `NeuralNetwork` (field 500) supporting `InnerProduct` and activations (`ReLU`, `Sigmoid`, `Tanh`), conforming `MLPClassifier` and `MLPRegressor` to `CoreMLExportable`.
4. **Binary Feature Scaler Export (`SwiftML`)**:
   - Added native binary `Scaler` Core ML protobuf encoding via `CoreMLExporter.exportBinaryStandardScaler` and `writeStandardScaler`.
5. **Numerical Stability & LAPACK Bounds (`SwiftCluster`)**:
   - Resolved LAPACK `dorgqr_` Householder reflector dimension violation in `RandomizedSVD` on rectangular matrices by clamping sketch dimension $l = \min(k + p, \min(M, N))$.
6. **Ecosystem Build Synchronization (`UkrainianNewsClassification`, `SwiftNLP`)**:
   - Conformed `MultinomialNaiveBayes` and `ComplementNaiveBayes` in `SwiftNLP` to `Codable`, fixing model serialization builds in downstream client pipelines.
7. **Dependency Graph Hardening (`Package.swift`)**:
   - Cleaned unused top-level dependency declaration `google/flatbuffers` from `Package.swift`.

---

### Version 3.3.0: Core ML Pipelines, Database TLS & VectorStore Foundation *(🟢 Completed)*

1. **End-to-End Core ML `Pipeline` Export (`SwiftML`, `SwiftPreprocessing`)**:
   - Chaining preprocessors (`StandardScaler`, `OneHotEncoder`) with estimators (`RandomForestClassifier`, `MLPClassifier`) into composite `PipelineClassifier` / `PipelineRegressor` Core ML messages.
   - Modern `.mlpackage` directory bundle serializer format.
   - FP16 ANE quantization flags (`quantizeFP16: true`).
2. **Enterprise Database Security & Writeback (`SwiftDatabase`)**:
   - Apple `Network.framework` (`NWConnection`) TLS/SSL socket handshakes for remote PostgreSQL and MySQL.
   - High-throughput batch `DataFrame.toSQL` bulk insert and upsert operations.
3. **In-Memory Vector Store & Local Text Embeddings (`SwiftCluster`, `SwiftNLP`)**:
   - In-memory `VectorStore` index supporting Cosine Similarity, Dot Product, and L2 distance.
   - Native `LocalEmbeddingEngine` for on-device RAG text embeddings.

---

### Version 3.4.0 – 3.5.0: Out-of-Core Data, Parquet Engine & Multimodal Perception *(🟢 Completed)*

1. **Out-of-Core & Large-Scale Data Processing (`SwiftDataFrame`)**:
   - `ChunkedDataFrame` and `LazyMemoryMappedCSVReader` for processing 100M+ row datasets exceeding RAM.
   - Pure-Swift zero-dependency Apache Parquet reader/writer with Snappy/Zstandard decompression.
   - Precomputed dictionary maps in `DataFrame._columns` and `Schema.fieldMap` for instantaneous $O(1)$ column resolution.
2. **On-Device LLMs & Quantized Execution (`SwiftLLM`)**:
   - 4-bit / 8-bit quantized execution (GGUF / AWQ) on MLX Metal for Llama-3, Qwen-2.5, Gemma-2.
   - Token-level JSON Schema constrained grammar decoding for local LLM output conforming to Swift `Codable`.
3. **Advanced Vision & Multimodal Perception (`SwiftVision`)**:
   - YOLOv8-Seg instance segmentation with proto mask heads.
   - CLIP-style vision-language feature matching on MLX Metal GPU.
   - Apple Accelerate hardware YOLO resizing via `vImageScale_PlanarF` with high-quality resampling.
4. **Multi-Agent Orchestration (`SwiftAgent`)**:
   - ReAct reasoning loops with multi-tool AST pipelines, dynamic backtracking, and normalized tool matching.
5. **Polynomial TreeSHAP (`SwiftExplain`)**:
   - Implemented $O(T \cdot L \cdot D^2)$ exact Shapley value algorithm on Data-Oriented Design flat tree node buffers.
6. **Spatial Indexing & Zero-Copy Clustering (`SwiftCluster`)**:
   - `KDTree` spatial index reducing `DBSCAN` neighbor search to $O(N \log N)$ and `VectorStore` zero-copy bounded heap search.
7. **Flat 1D Accelerate Vectorization (`SwiftPreprocessing`, `SwiftForecast`)**:
   - Migrated scalers and Kalman filter matrices to contiguous 1D row-major buffers vectorized via `vDSP_vsubD`, `vDSP_vsdivD`, and `vDSP.sort`.
8. **Swift 6 Actor-Isolated Database Drivers (`SwiftDatabase`)**:
   - Migrated `SQLiteConnection`, `PostgreSQLConnection`, and `MySQLConnection` to `public actor` types for complete data-race freedom.
9. **Nelder-Mead Simplex Optimizer (`SwiftForecast`)**:
   - Derivative-free simplex optimization for automatic MLE fitting of parameters in exponential smoothing models.
10. **Sequential AutoML Engine (`SwiftOptimize`)**:
    - Automated model selection with 3-fold cross-validation.
11. **Native SwiftUI 2D Charts (`SwiftVisualization`)**:
    - Enhanced `SwiftSciChartView` with 2D Heatmaps, Scatter plots, and Histograms.

---

### Version 3.5.1: Pure-Swift Computer Vision, Anomaly Detection, Out-of-Core Joins & Local Agents *(🟢 Completed)*

Targeted architectural enhancements resolving open issues from the ecosystem audit:

1. **Pure-Swift Non-Maximum Suppression (`NonMaximumSuppression`, `SwiftVision`)**:
   - Implemented 100% native pure-Swift NMS algorithm (`NonMaximumSuppression.filter`) with `BoundingBox.area` and `BoundingBox.intersectionOverUnion(with:)` (IoU), completely eliminating `torchvision.ops.nms` and OpenCV dependencies.
2. **Seasonal ESD Time Series Anomaly Detection (`TimeSeriesAnomalyDetector`, `SwiftForecast`)**:
   - Implemented native statistical anomaly detection (`TimeSeriesAnomalyDetector`, `TimeSeriesAnomaly`, `AnomalyDetectionResult`) based on Seasonal Hybrid ESD (S-ESD) and Median Absolute Deviation (MAD), fully resolving G-008 without `statsmodels`.
3. **Out-of-Core Hash Join on Chunked DataFrames (`ChunkedDataFrame.join`, `SwiftDataFrame`)**:
   - Added streaming relational `join(_:on:how:)` supporting inner, left, and outer join semantics without full dataset in-memory allocations.
4. **95% Confidence Prediction Bounds (`ExponentialSmoothing`, `SwiftForecast`)**:
   - Extended `ForecastResult` with analytical 95% uncertainty intervals (`lowerBound`, `upperBound`) computed via residual standard error expansion.
5. **Unified `LLMModel` Protocol Conformance (`TransformerDecoder`, `SwiftLLM`)**:
   - Conformed `TransformerDecoder` to the public `LLMModel` protocol for uniform streaming token generation across models.
6. **Local Native LLM Reasoning Overload (`ReActAgent`, `SwiftAgent`)**:
   - Added `run(query:model:options:)` overload executing autonomous ReAct agent reasoning loops directly on Apple Silicon without network latency or external APIs.
7. **Lineage Audit Trail & Syntax Extensions (`SwiftAgentEvaluator`, `SwiftAgent`)**:
   - Added support for parameter-labeled AST syntax (`filter(column:condition:)`, `select(columns:)`, `sample(n:)`, `head(n:)`, `tail(n:)`) and transformation lineage tracking.
8. **100% DocC Public API Documentation**:
   - Maintained 100.00% public documentation coverage across all 1,537 API symbols.

---

### Version 3.5.2: Standard Apache Parquet, Pure-Swift NumPy Reader, SQLite Auto-Discovery & Quantile Regression *(🟢 Completed)*

Critical enhancements enabling zero-dependency reading of industry-standard scientific formats and non-parametric model inference:

1. **Standard Apache Parquet Compliance (`ParquetReader`, `SwiftDataFrame`)**:
   - Complete pure-Swift support for industry-standard Apache Parquet files generated by PyArrow, DuckDB, Pandas, and Hugging Face Hub.
   - Parses uncompressed Thrift `PageHeader` at each page offset, handles dictionary pages (`DICTIONARY_PAGE`), decompresses Snappy payload slices per page, unpacks dynamic bit-width `RLE_DICTIONARY` and `PLAIN_DICTIONARY`, decodes multi-level repetition/definition levels, and aggregates repeated nested list schemas (`list<item>`). Bit-exact verified on Hugging Face `go_emotions` (5,427 rows).
2. **Pure-Swift NumPy NPY & NPZ Multi-Dimensional Tensor Reader (`NPYReader`, `NPZReader`, `SwiftDataFrame`)**:
   - Native zero-dependency reader for `.npy` files and `.npz` archive containers. Supports little-endian multidimensional tensors, ZIP64 extended records (`0x0001`), and Deflate decompression via macOS `Compression` framework. Added convenience initializers `DataFrame(npy:)` and `DataFrame(npz:)`.
3. **Automatic SQLite Table Discovery (`DataFrame(sqlite:)`, `SwiftDatabase`)**:
   - Added `DataFrame(sqlite: URL, table: String? = nil)` in `SwiftDatabase`, auto-discovering user tables from `sqlite_master` and mapping SQL query results directly into typed DataFrame columns without manual SQL boilerplate.
4. **Quantile Regression / Pinball Loss in GBDT (`GBDTLoss`, `GradientBoostedTreesRegressor`, `SwiftML`)**:
   - Added `GBDTLoss` (`.squaredError`, `.absoluteError`, `.quantile(alpha:)`) to `GradientBoostedTreesRegressor`. Calculates asymmetric pinball negative gradients and per-leaf optimal quantile estimates, allowing automated 80% / 95% non-parametric confidence bands.
5. **Extended CLI Format Support (`SwiftSciCLI`)**:
   - Enhanced `swiftsci summary` and `swiftsci convert` to inspect and convert `.parquet`, `.npy`, and `.npz` files alongside CSV and Feather.
6. **Scientific Multi-Round Benchmark Harness (`SwiftSciBenchmarks`)**:
   - Robust statistical runner reporting 95% confidence intervals, trimmed mean, and live Mach RSS memory tracking across 40+ performance benchmarks.

---

### Version 3.6.0: Next-Generation Scaling, Determinism, Memory Safety & Algorithmic Parity *(🟢 Completed)*

*Detailed implementation plan:* [`ROADMAP/implementation_plan_36.md`](../implementation_plan_36.md)  
*Primary Focus:* Resolving memory-safety invariants under concurrency, strict Scikit-Learn mathematical parity, and next-generation large-scale structures across 4 prioritized sprints.

#### 🔴 Sprint 1: P0 — Critical Memory & Concurrency Safety

1. **ARC Retention of Backing Arrow Buffer (`ArrowDataBuffer`, `SwiftDataFrame`)**:
   - Add strong reference `owner: AnyObject?` to `ArrowDataBuffer` to retain the backing `ArrowTable` or `ArrowArray`.
   - Guarantees zero use-after-free when buffer slices cross asynchronous task boundaries in Swift Concurrency.
2. **Structured GPU Memory Cleanup & `WiredMemoryTicket` Safety (`SwiftPreprocessing`, `SwiftLLM`)**:
   - Remove unstructured `Task { MLX.Memory.clearCache() }` from `WiredMemoryTicket.deinit`.
   - Introduce scoped closure API `withMemoryTicket` ensuring Apple Silicon GPU graphs are evaluated (`MLX.eval()`) prior to cache clearance, preventing out-of-order GPU memory corruption.
3. **Value Semantics & Data Leakage Prevention in `Pipeline` (`SwiftPreprocessing`, `SwiftOptimize`)**:
   - Extend `PreprocessingTransformer` protocol with `copyTransformer() -> any PreprocessingTransformer`.
   - Enforce deep-copy value semantics in `Pipeline` so concurrent cross-validation folds (`TaskGroup`) operate on strictly isolated transformer states without state contamination.

#### 🟠 Sprint 2: P1 — Mathematical Robustness & Algorithmic Parity

4. **Moore-Penrose Pseudo-Inverse Fallback in `KalmanFilter` (`SwiftForecast`)**:
   - Implement SVD-based Moore-Penrose pseudo-inverse via LAPACK `dgesdd_`: $S^+ = V \cdot \Sigma^+ \cdot U^T$.
   - Replaces naive LU decomposition for innovation covariance $S = H P H^T + R$ when $S$ is singular or ill-conditioned ($R = 0$), preventing fatal `singularMatrix` crashes.
5. **Deterministic PCA Axis Orientation via `svd_flip` (`SwiftCluster`)**:
   - Implement `svd_flip` forcing positive signs on singular vector elements with maximum absolute magnitude in `PCA` and `RandomizedSVD`.
   - Guarantees 100% deterministic axis parity with `sklearn.decomposition.PCA`.
6. **Numerically Stable Variance via Welford's Algorithm & Strict `checkNaN` (`SwiftStats`)**:
   - Replace naive $E[X^2] - (E[X])^2$ formula in `Stats.variance` with two-pass centered deviation accumulation using Accelerate `vDSP_vsubD` and `vDSP_measqvD`.
   - Change default parameter to `checkNaN: true` in `Stats.describe` to eliminate silent NaN propagation.
7. **Unified Missing Value Semantics & `NullStrategy` (`SwiftDataFrame`)**:
   - Introduce internal `validityBitmap: [UInt8]?` in `TypedColumn` for fast bitmask missingness queries.
   - Provide explicit `NullStrategy` (`.nan`, `.drop`, `.zero`) when exporting columns to contiguous Accelerate/MLX feature matrices (`toFeatureMatrix`).

#### 🟡 Sprint 3: P2 — Security, PRNG & Robustness

8. **Zero-Division Defense in `KernelSHAP` (`SwiftExplain`)**:
   - Guard boundary coalition weights against $k(M - k) = 0$ division.
   - Implement exact analytical Shapley calculation for low dimensions ($M \le 2$) without randomized sampling.
9. **`handleUnknown: .ignore` in `OneHotEncoder` (`SwiftPreprocessing`)**:
   - Add `HandleUnknownStrategy` (`.error`, `.ignore`). Unseen categories at inference time produce all-zero vectors instead of throwing runtime exceptions.
10. **Reversible Byte-Level UTF-8 Decoder in `BPETokenizer` (`SwiftNLP`)**:
    - Implement reverse mapping `byteDecoder: [Character: UInt8]`.
    - Reconstruct byte arrays `[UInt8]` prior to `String(decoding:as: UTF8.self)`, ensuring intact reconstruction of Cyrillic, CJK, and compound emojis.
11. **XSS Prevention & HTML Sanitization in `SwiftVisualization` (`SwiftVisualization`)**:
    - Replace raw string interpolation with `JSONEncoder` for label arrays and apply HTML entity escaping to `<title>` and annotations in interactive Plotly exports.
12. **Execution Timeouts in `ReActAgent` (`SwiftAgent`)**:
    - Guard tool invocations with `toolTimeoutSeconds: Double = 30.0` via `withThrowingTaskGroup`, preventing unresponsive tools from locking the reasoning loop.
13. **Comprehensive Transformation Lineage Audit Trail in `SwiftAgentEvaluator` (`SwiftAgent`)**:
    - Introduce immutable `LineageRecord` structures tracking step index, operation type, input/output row counts, and timestamps.
14. **Zero-Heap Bounding Boxes via `BoundingBoxSIMD` (`SwiftVision`)**:
    - Introduce `BoundingBoxSIMD` using flat `SIMD4<Float>` coordinates and integer `classId`, eliminating heap allocations and ARC retain/release overhead during NMS filtering.
15. **Native Database Type Widening in `DatabaseConnection` (`SwiftDatabase`)**:
    - Extend `AnySendableValue` with native `.int64`, `.bool`, `.date`, `.data` representations and add direct SQLite column BLOB reading.
16. **High-Quality Xoshiro256++ PRNG & `randomState` Propagation (`SwiftPreprocessing`, `SwiftML`, `SwiftCluster`)**:
    - Replace linear congruential generator with Xoshiro256++ (period $2^{256} - 1$, passes BigCrush).
    - Expose `randomState: Int?` across `RandomForestClassifier`, `RandomForestRegressor`, `KMeans`, and `PCA` for reproducible experiments.

#### 🚀 Sprint 4: Features & Next-Gen Scaling (v3.6.0 Core)

17. **HNSW Approximate Nearest Neighbor Graph Index (`SwiftCluster`)**:
    - Hierarchical Navigable Small World (`HNSWIndex`) graph search over high-dimensional vector embeddings, delivering $O(\log N)$ sub-millisecond retrieval on collections $>100,000$ vectors.
18. **256-Bin Histogram Quantization in GBDT (`SwiftML`)**:
    - Fast `HistGradientBoostingClassifier` discretizing continuous features into `UInt8` bins.
    - Accelerates split search from $O(N \log N)$ to $O(K)$ via single-pass gradient/hessian histogram accumulation (LightGBM-style).
19. **Concurrent Cross-Validation & Multi-Class Training (`SwiftOptimize`, `SwiftML`)**:
    - Concurrently evaluate cross-validation folds in `AutoML` and train per-class binary estimators in `OneVsRestClassifier` using `withThrowingTaskGroup`.
20. **Early Stopping Callback with `patience` (`SwiftML`)**:
    - Add `EarlyStopping` callback (`patience`, `minDelta`, `restoreBestWeights`) for `MLPClassifier`, `MLPRegressor`, and `GradientBoostedTrees`.
21. **Compressed Sparse Matrix Storage — CSR / CSC (`SwiftPreprocessing`, `SwiftNLP`)**:
    - Implement `SparseMatrix<T>` with Apple Accelerate Sparse BLAS routines (`sparse_matrix_vector_multiply`), reducing RAM usage $10\times-50\times$ for high-cardinality one-hot encodings and TF-IDF vocabularies.
22. **Concurrent Order Search in `AutoARIMA` / `SARIMA` (`SwiftForecast`)**:
    - Parallelize hyperparameter grid exploration across $(p, d, q) \times (P, D, Q)_s$ via `TaskGroup` evaluated against AIC/BIC criteria.
23. **Native Metal MSL SIMD Kernels for Q4/Q8 Quantization (`SwiftLLM`)**:
    - Custom Metal Shading Language compute kernels (`QuantizedGEMM.metal`) using `simdgroup_matrix` for hardware-accelerated 4-bit/8-bit dequantization and matrix multiply on Apple Silicon GPUs.
24. **Multi-Agent Orchestration & Communication Channels (`SwiftAgent`)**:
    - Asynchronous message bus (`AgentMessageBus` on `AsyncStream`) orchestrating specialized agents (Analyst, Planner, Critic) with parallel tool dispatch.
25. **Batch-Buffered SQLite Ingestion & SCRAM-SHA-256 Authentication (`SwiftDatabase`)**:
    - Buffer SQLite column reads in 1024-row batches and implement standard SCRAM-SHA-256 password authentication via Apple CryptoKit for PostgreSQL.

#### 🧪 Verification & Quality Gate Matrix (v3.6.0)

1. **Compiler Diagnostics**: Zero errors and zero warnings (`-warnings-as-errors`) under Swift 6 Complete Concurrency Checking.
2. **Swift Testing Coverage**: 100% regression pass rate across all 14 modules with new edge-case tests (singular covariance matrices, $M \le 2$ SHAP, unseen categories).
3. **Mathematical Parity Gate**: Verification against Scikit-Learn / SciPy / FilterPy: $\max |y_{\text{Swift}} - y_{\text{Python}}| < 10^{-4}$; `svd_flip` component orientation parity $< 10^{-6}$.
4. **100% DocC API Compliance**: Clean documentation generation with zero DocC warnings via `swift package generate-documentation`.

---

### Version 3.6.1: WordNet Lexicon Expansion, Princeton Ingestion & Database Health Protocols *(🟢 Completed)*

*Detailed implementation plan:* [`ROADMAP/implementation_plan_36_1.md`](../implementation_plan_36_1.md)  
*Primary Focus:* Expanding semantic NLP coverage with a comprehensive core WordNet taxonomy and standard file loader, and introducing connectivity health checks / endpoint symmetry for relational database drivers.

1. **Comprehensive Offline WordNet Taxonomy (`SwiftNLP`)**:
   - Expanded default synsets baseline to ~120 core synsets spanning all major POS categories, physical entities, living organisms, human roles, artifacts, computing, abstract concepts, actions, and qualitative attributes.
2. **Princeton WordNet File Loader & Tokenizer (`SwiftNLP`)**:
   - Implemented pure-Swift loader (`WordNet.load(fromDataFile:pos:)`, `WordNet.load(fromDirectory:)`) capable of ingesting official Princeton WordNet distribution files (`data.noun`, `data.verb`, `data.adj`, `data.adv`).
3. **Database Connectivity Health Protocol (`SwiftDatabase`)**:
   - Added `ping() async -> Bool` to `DatabaseConnection` protocol with zero-allocation test queries for SQLite, PostgreSQL, and MySQL.
4. **Driver Symmetrical Initialization (`SwiftDatabase`)**:
   - Introduced convenience initializers with explicit endpoint parameters (`host:port:user:password:database:sslMode:`) for `MySQLConnection` and `PostgreSQLConnection`.
5. **DocC Documentation Parity**:
   - Fully aligned DocC catalogs with pure-Swift wire-protocol driver capabilities.


---

## 🏛 Integration Guidelines for Client Applications

Thanks to its modular design, SwiftSci seamlessly integrates into applications following clean architecture:

* **View Models:** All model initialization, dataset loading (`SwiftDataFrame`), and preprocessing pipeline configurations reside in the View Model layer.
* **Background Tasks:** Method calls like `.fit()` for compute-heavy algorithms (e.g. Random Forest or `MLX` graph evaluations) should be executed inside isolated background tasks (`Task.detached { }`) to prevent main thread blocking and maintain smooth 120Hz UI rendering.
* **Complexity Encapsulation:** Low-level Arrow memory buffers and non-Sendable `MLXArray` handles are encapsulated as `internal`. Client applications interact strictly with thread-safe, public Swift 6 API contracts.


