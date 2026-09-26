# SwiftSci 3.10.1

**SwiftSci** is a native, high-performance, modular scientific computing and machine learning library for Swift. Built from the ground up for Apple Silicon (M-series) Unified Memory Architecture (UMA), SwiftSci is fully compliant with Swift 6 strict concurrency requirements.

It seamlessly blends hardware-accelerated tensor operations on the Apple Silicon GPU via **MLX** with ultra-optimized CPU vector routines from the **Accelerate framework (vDSP / LAPACK / BLAS)** and Apple's native **NaturalLanguage** framework.

[![Swift Version](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/Nodibell/SwiftSci)
[![Platform Compatibility](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/Nodibell/SwiftSci)
[![codecov](https://codecov.io/gh/Nodibell/SwiftSci/graph/badge.svg)](https://codecov.io/gh/Nodibell/SwiftSci)
[![Documentation Coverage](https://img.shields.io/badge/DocC%20Coverage-100%25-brightgreen)](https://nodibell.github.io/SwiftSci/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## ⚡ Key Architectural Advantages

- 🚀 **Apple Silicon Native (UMA)**: Zero-copy tensor and columnar memory sharing between CPU and Metal GPU via MLX.
- 🛡️ **Swift 6 Strict Concurrency**: 100% data-race free with native Actors and Sendable isolation.
- 📦 **14 Integrated Modules**: Full end-to-end data stack from Apache Parquet & Arrow DataFrames to GBDT, Time Series ARIMA, XAI (TreeSHAP/LIME), Local LLMs, Computer Vision, and Autonomous ReAct Agents.
- 📉 **Radical Memory Efficiency**: Consumes **10× to 70× less RAM** than Python runtime stacks (Pandas, NumPy, Scikit-Learn, PyTorch).
- 🧩 **Zero-Boilerplate Ingestion**: Pure-Swift readers for Apache Parquet, Apache Arrow Feather, NumPy `.npy`/`.npz`, and automatic SQLite schema discovery.

---

## 📦 Installation

Add SwiftSci to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/Nodibell/SwiftSci.git", from: "3.10.1")
]
```

Or add individual targeted modules to your target dependencies:

```swift
.target(
    name: "MyApp",
    dependencies: [
        .product(name: "SwiftDataFrame", package: "SwiftSci"),
        .product(name: "SwiftML", package: "SwiftSci"),
        .product(name: "SwiftForecast", package: "SwiftSci"),
    ]
)
```

> **Platform Requirement:** macOS 14+ (Apple Silicon M-Series).

---

## 💻 Quick Start

```swift
import Foundation
import SwiftDataFrame
import SwiftML
import SwiftPreprocessing
import SwiftExplain
import SwiftNLP

// 1. Columnar DataFrame with zero-copy SIMD operations
let xCol = TypedColumn<Double>(name: "feature", values: [1.0, 2.0, 3.0, 4.0, 5.0])
let yCol = TypedColumn<Double>(name: "target", values: [2.1, 3.9, 6.2, 8.0, 10.1])
let df = try DataFrame(columns: [xCol, yCol])

// 2. High-speed Machine Learning (Linear Regression / GBDT / RF)
let regressor = LinearRegression()
let X = try df.toFeatureMatrix(columnNames: ["feature"])
let y = try df.toTargetVector(columnName: "target")
try await regressor.fit(features: X, targets: y)

// 3. Fast Categorical OneHotEncoder (5× faster than Scikit-Learn)
let ohe = OneHotEncoder()
ohe.fit([["engineering"], ["finance"], ["engineering"]])
let encoded = try ohe.transform([["engineering"], ["finance"]])

// 4. Black-Box Model Explainability (TreeSHAP & LIME)
let lime = LIMEExplainer(kernelWidth: 0.75, regularization: 0.01)
let explanation = await lime.explain(model: { $0.reduce(0.0, +) }, instance: [1.0, 2.0, 3.0], numSamples: 300)

// 5. Native Sentiment Analysis
let vader = VADERSentimentAnalyzer()
let score = vader.polarityScores(text: "SwiftSci 3.8.0 is incredibly fast, memory efficient, and robust!")
print("Sentiment compound score:", score.compound)
```

---

## 🧩 The 14 Core Modules

| Module | Purpose & Capabilities | Documentation |
| :--- | :--- | :---: |
| **`SwiftDataFrame`** | Zero-copy columnar tables, **pure-Swift Apache Parquet engine**, Arrow Feather with ARC lifetime safety, NumPy `.npy`/`.npz` tensor reader, memory-mapped I/O, SIMD hash joins, out-of-core `ChunkedDataFrame`, and `ArrowNullStrategy`. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftdataframe/) |
| **`SwiftStats`** | Numerically stable two-pass variance via Accelerate vDSP, SIMD sorting, Student-t/Chi-Square/F distributions, Two-Sample t-test, Spearman correlation, ANOVA. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftstats/) |
| **`SwiftPreprocessing`** | Feature scaling (`StandardScaler`, `MinMaxScaler`), categorical encoding (**`OneHotEncoder`** with `handleUnknown: .ignore`, `TargetEncoder`), **`SparseMatrix`** (CSR/CSC) with Apple Accelerate Sparse BLAS, `Pipeline` with deep-copy value semantics, `ColumnTransformer`, `HardwareRouter`. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftpreprocessing/) |
| **`SwiftML`** | OLS/Ridge/Lasso regression, Decision Trees, Random Forests, **HistGradientBoosting** (256 bins), GBDT Quantile Regression, **EarlyStopping**, **`LinearSVC`** (Metal GPU), MLP neural nets, Core ML & ONNX exporters, reproducible Xoshiro256++ PRNG. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftml/) |
| **`SwiftCluster`** | **`HNSWIndex`** graph index for $O(\log N)$ ANN search, in-memory **`VectorStore`** cosine index, randomized SVD / PCA with deterministic `svdFlip`, DBSCAN, `IsolationForest` outlier detection, `KMeans`. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftcluster/) |
| **`SwiftForecast`** | **`AutoARIMA`** parallel order search, ARIMA/SARIMA, Exponential Smoothing with 95% confidence intervals, ETS, GARCH, Kalman filtering with Moore-Penrose pseudo-inverse fallback, Prophet-style trend decomposition, Seasonal ESD anomaly detection. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftforecast/) |
| **`SwiftOptimize`** | Concurrent cross-validation via `TaskGroup`, parallel **`AutoML`**, Forecast Errors Suite (RMSE, MAE, MAPE, R²), ROC-AUC / PR-AUC metrics, `GridSearchCV`. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftoptimize/) |
| **`SwiftExplain`** | Model interpretability: parallel `KernelSHAP` with zero-division guards and exact paths for $M \le 2$, exact polynomial **`TreeSHAP`**, **`LIMEExplainer`**, Partial Dependence Plots (PDP), Permutation Importance. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftexplain/) |
| **`SwiftNLP`** | Linguistic engine: byte-level UTF-8 invertible **`BPETokenizer`**, lemmatization, Porter stemmer, POS tagger, **`VADERSentimentAnalyzer`**, `NaiveBayesClassifier`, dense embeddings. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftnlp/) |
| **`SwiftLLM`** | **`LLMModel` protocol**, custom Metal MSL SIMD-group quantization kernels (`QuantizedGEMM.metal`), 4-bit/8-bit quantized linear layers (`QuantizedLinear`), Paged KV-Cache allocator, constrained JSON grammar decoder, streaming generation. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftllm/) |
| **`SwiftVision`** | Zero-heap hardware-accelerated **`BoundingBoxSIMD`** NMS, YOLOv8n object detection, YOLOv8-Seg instance segmentation, CLIP projector, U-Net image segmentation. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftvision/) |
| **`SwiftDatabase`** | Zero-copy SQL ingestion/export: SQLite with **1024-row column buffer ingestion** and auto-discovery, native PostgreSQL with **RFC 5802/7677 SCRAM-SHA-256** auth, MySQL connectors, and strongly-typed `AnySendableValue`. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftdatabase/) |
| **`SwiftAgent`** | Omni-module autonomous agents: **`SwiftSciToolbox`** scientific tools, typed **`AgentToolV2`** with JSON schemas, **`ReActAgent`** with real-time async event streaming, **`SemanticVectorMemory`** (HNSW graph recall), and native SwiftUI **`AgentDialogueController`** (`@Observable`). | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftagent/) |
| **`SwiftVisualization`** | Native SwiftUI `Canvas` interactive charts (`SwiftSciChartView`), XSS-sanitized Plotly HTML export, and terminal ASCII/Braille graphs. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftvisualization/) |

---

## 📊 Performance & Memory Comparison (SwiftSci 3.10.1 vs Python)

The following benchmark metrics reflect the certified, dual-language automated test harness run on Apple Silicon (M-series, macOS 15, IEEE 754 64-bit Double Precision, Swift 6 Release vs Python 3.11 with Pandas, NumPy, Scikit-Learn, and Statsmodels). All workloads execute on byte-identical shared fixtures with synchronized hyperparameters under **Benchmark Methodology v2**:

> [!NOTE]
> SwiftSci and the Python reference stack use different implementation paths and optimization strategies. The measured difference therefore reflects both algorithmic implementation and library/runtime overhead, rather than Python interpreter overhead alone.

| Domain / Scenario | SwiftSci 3.10.1 | Python Reference Stack | Relative Performance | Scope / Implementation |
| :--- | :---: | :---: | :---: | :--- |
| **Holt-Winters Fit** (50k pts, s=12) | **`16.11 ms`** | `3,431.44 ms` (*Statsmodels*) | **SwiftSci 212.97×** | Native Nelder-Mead simplex optimizer |
| **ARIMA(1,1,1) Fit** (50k pts) | **`2.26 ms`** | `594.36 ms` (*Statsmodels*) | **SwiftSci 262.49×** | Exact Gaussian likelihood solver |
| **NaiveBayes Fit** (1k×100, 3 classes) | **`0.027 ms`** | `0.410 ms` (*Scikit-Learn*) | **SwiftSci 15.35×** | Accelerate `vDSP_dotprD` SIMD dot-product |
| **SQLite DataFrame Ingestion** | **`0.025 ms`** | `0.443 ms` (*Pandas*) | **SwiftSci 17.90×** | Persistent in-memory C-API (63× less RAM) |
| **RandomForest Fit** (1k×4, 50 trees, d=4, gini) | **`4.13 ms`** | `31.23 ms` (*Scikit-Learn*) | **SwiftSci 7.56×** | Data-Oriented Design (DOD) tree buffers |
| **GBDT Regressor Fit** (1k×4, 50 est.) | **`9.09 ms`** | `32.82 ms` (*Scikit-Learn*) | **SwiftSci 3.61×** | Contiguous gradient-boosted ensembles |
| **OneHotEncoder** (50k rows) | **`5.15 ms`** | `31.51 ms` (*Scikit-Learn*) | **SwiftSci 6.12×** | SIMD categorical bitmask transform |
| **Classification ROC-AUC** (50k preds) | **`2.65 ms`** | `7.60 ms` (*Scikit-Learn*) | **SwiftSci 2.87×** | Single-pass sorted trapezoidal integration |
| **KernelSHAP Explain** (100 coalitions) | **`0.189 ms`** | `0.434 ms` (*SHAP*) | **SwiftSci 2.30×** | Zero-division guarded coalitional sampling |
| **Two-Sample T-Test** (100k samples) | **`0.263 ms`** | `0.411 ms` (*SciPy*) | **SwiftSci 1.56×** | Accelerate vDSP Welch's t-test |
| **Pearson Correlation** (500k pairs) | **`0.828 ms`** | `1.209 ms` (*NumPy*) | **SwiftSci 1.46×** | SIMD dot-product covariance |
| **Mean Reduction** (vDSP 1M doubles) | **`0.081 ms`** | `0.119 ms` (*NumPy*) | **SwiftSci 1.46×** | Apple Accelerate `vDSP_meanvD` |
| **CSV Read** (100k rows) | **`18.17 ms`** | `19.53 ms` (*Pandas*) | **SwiftSci 1.08×** | POSIX `mmap` zero-copy chunk parsing |
| **LinearSVC Fit** (1k×4, 100 epochs, Metal GPU) | **`0.404 ms`** | `0.384 ms` (*Scikit-Learn*) | **Python 1.05×** | Apple Metal GPU kernel vs LibLinear C |
| **PCA SVD fitTransform** (1k×100 → 10 comps) | **`1.13 ms`** | `0.758 ms` (*Scikit-Learn*) | **Python 1.49×** | Accelerate LAPACK `dgesdd_` vs SciPy BLAS |
| **KMeans Fit** (10k×4, 3 clusters, 50 iters) | **`17.44 ms`** | `7.31 ms` (*Scikit-Learn*) | **Python 2.39×** | Underflow-clamped SIMD distance vs Cython k-means |
| **SortBy Double Column** (100k rows) | **`44.84 ms`** | `7.49 ms` (*NumPy*) | **Python 5.99×** | Swift sort vs NumPy quicksort in C |
| **DataFrame SIMD Hash Join** (100k rows) | **`35.20 ms`** | `0.456 ms` (*Pandas*) | **Python 77.19×** | Swift typed hash table vs Pandas C hashtable |
| **SwiftLLM Incremental Decode** (RoPE + KV-Cache) | **`0.125 ms`** | `1.450 ms` (*Full Forward*) | **SwiftSci 11.60×** | $O(1)$ single-token decode ($Q \times K/V$), $\Delta \le 1.03 \times 10^{-7}$ |
| **Metal MSL gemv_q4_0 Kernel** (1024d) | **`0.015 ms`** | `0.045 ms` (*Float32 Dequant*) | **SwiftSci 3.00×** | Zero-copy packed Q4_0 evaluation |
| **AutoARIMA Zero-Variance Guard** (1000 pts) | **`< 0.01 ms`** | Divergent loop | **Instant Exit** | Guaranteed non-diverging guard |

### 🎯 Model Accuracy & Forecast Quality Scorecard

SwiftSci incorporates an automated validation scorecard confirming numerical parity against ground truth test datasets:

```text
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │                    MODEL ACCURACY & FORECAST QUALITY SCORECARD                     │
  ├────────────────────────────────────────────────────────────────────────────────────┤
  │ [Forecast] Holt-Winters (h=24) : RMSE=0.350, MAE=0.281, MAPE=0.21%, R²=0.997       │
  │ [Forecast] ARIMA(1,1,1) (h=24) : RMSE=10.218, MAE=8.557, MAPE=5.87%, R²=-1.555     │
  │ [ML Reg]   GBDT (30 trees, d=4) : RMSE=0.421, MAE=0.344, R²=0.9879                 │
  │ [ML Cls]   RandomForest (30 tr.): Accuracy=98.50%, F1=0.986, ROC-AUC=0.999         │
  │ [NLP Cls]  NaiveBayes (3-class) : Accuracy=35.00%, Macro-F1=0.342                  │
  └────────────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 What's New in v3.10.1

- **Descending Sort with Missing Values (`SwiftDataFrame`)**: Preserved descending order for non-null values while keeping null elements sorted last, working around a Swift 6 compiler closure constraint with `any Comparable` (PR #38).
- **Benchmark Methodology v2 (`Benchmarks`)**: Byte-identical deterministic IEEE-754 `.bin` fixtures with SHA-256 validation, synchronized hyperparameters across Swift and Python runners, and 3-tier categorization with `Relative Performance` metrics.
- **GBDT Benchmark Dimension Fix (`SwiftML`)**: Fixed feature and target dimension alignment on 1k×4 fixtures, measuring real wall-clock fit (9.09 ms in Swift vs 32.82 ms in Scikit-Learn, 3.61× speedup).
- **Streamlined Repository & Documentation**: Removed legacy presentation slides and artifacts.

---

## 🚀 What's New in v3.10.0

- **Verified Native Local LLM Runtime Pipeline (`SwiftLLM`)**:
  - `SamplingConfiguration`: Fully decoupled sampling configuration (temperature, topK, topP, repetitionPenalty) operating strictly at the logits level prior to softmax.
  - `RoPEEmbedding`: Rotary Positional Embeddings with dynamic `positionOffset` support for incremental token decoding.
  - `TransformerDecoder`: Two-stage inference pipeline with full prompt prefill and incremental single-token decoding with accumulated KV-cache.
  - `QuantizedTensor` & Metal GEMV: Direct retention of packed Q4_0 and Q8_0 weights from GGUF into native buffers without artificial dequantization; verified parity against Apple Metal MSL GEMV kernels (`gemv_q4_0`, `gemv_q8_0`).
  - `MLX.compile`: Integrated single-token decode graph compilation with exact parity verification against uncompiled execution.
- **Resilient Time-Series Forecasting (`SwiftForecast`)**:
  - `AutoARIMA`: Added zero-variance detection for constant series and enforced `maxIterations = 500` loop termination cap, preventing infinite optimization loops.
- **Chat Template Integration Layer (`SwiftNLP`)**:
  - `ChatMessage` & `ChatTemplate`: Native formatting for modern LLM chat formats (Llama-3 `<|begin_of_text|>...<|eot_id|>`, ChatML `<|im_start|>...<|im_end|>`, Mistral `[INST]...[/INST]`) with direct `encode` tokenization integration.
- **Robust Autonomous Agents (`SwiftAgent`)**:
  - `AgentParameterSchema`: JSON schema validation and structured parsing (`validate(arguments:)`, `parseAndValidate(jsonString:)`).
  - `ReActAgent`: Resilient error handling feeding malformed tool inputs back into the reasoning trajectory for self-correction without crashing the agent loop.
- **Hardened C-Pointer Memory Safety (`SwiftDatabase`)**:
  - SQLite zero-copy connector upgraded to `sqlite3_close_v2`, explicit `close()` lifecycle, and hardened statement finalization, verified clean under AddressSanitizer.

## 🚀 What's New in v3.9.0

- **Advanced NLP Vectorization & Text Cleaning (`SwiftNLP`)**:
  - `TFIDFVectorizer`: Support for configurable n-gram extraction (`ngramRange: 1...2` / `1...3`), sublinear term frequency scaling ($1 + \log(\text{tf})$), and maximum document frequency pruning (`maxDF: 0.8`).
  - `TextCleaner`: Fast scalar-based regex normalizer stripping HTML tags, emails, URLs, special punctuation, and excess whitespace.
- **Stratification & Advanced Imbalanced Resampling (`SwiftPreprocessing`)**:
  - `trainTestSplit`: Added `stratify` parameter preserving exact multi-class proportions across train and test partitions.
  - `DataFrame.trainTestSplit`: Added `stratifyColumn` for stratified tabular splits.
  - `StratifiedKFold`: K-Fold split generator with strict per-fold class frequency preservation.
  - `BorderlineSMOTE`: Decision-boundary danger-zone synthetic oversampling.
  - `ADASYN`: Adaptive density-weighted synthetic minority oversampling.
  - `DataFrame.resample`: Resampling pipeline supporting SMOTE, BorderlineSMOTE, ADASYN, and RandomUndersampler with automatic categorical string target encoding/decoding.
- **Manifold Learning & High-Dimensional Projection (`SwiftCluster`)**:
  - `TSNE`: Exact t-Distributed Stochastic Neighbor Embedding with binary search for perplexity entropy, early exaggeration, adaptive momentum, and KL-divergence reporting.
  - `DataFrame.tsne`: Direct dimensionality reduction on numerical DataFrame columns producing low-dimensional coordinates (`tsne_1`, `tsne_2`).
- **Model Interpretability & Feature Importance (`SwiftML`)**:
  - `OneVsRestClassifier` & `LinearSVCOneVsRest`: Added `topFeatures(classIndex:topN:vocabulary:)`, `topFeatures(classIndex:topN:featureNames:)`, and `topFeaturesPerClass` to inspect the most predictive tokens and feature weights per category.
  - `FeatureImportance`: Standardized Sendable/Codable representation of feature weights and indices.
- **Exploratory Data Analysis (EDA) Interactive Charts (`SwiftVisualization`)**:
  - `ChartExporter.plotClassDistribution` & `DataFrame.plotClassDistribution`: Interactive Plotly bar chart with class frequencies and percentage labels.
  - `ChartExporter.plotBoxPlot` & `DataFrame.plotBoxPlot`: Interactive box plots with quartile statistics, medians, and outlier detection.

---

## 🚀 What's New in v3.8.2

- **Multi-Page Apache Parquet Stream Decoding (`SwiftDataFrame`, `SwiftNLP`)**: Full multi-page stream decoding in `ParquetReader` supporting Snappy decompression across concatenated data and dictionary pages; added direct path ingestion `DataFrame.readParquet(_ path: String)`.
- **Native Core ML Bundle Export (`SwiftML`)**: Automated `.mlpackage` directory export via `CoreMLExporter.export(...)` generating schema-complete metadata and feature descriptions for native Apple Silicon deployment.
- **Non-Throwing Random Forest Initializers (`SwiftML`)**: Streamlined `RandomForestClassifier` and `RandomForestRegressor` initialization with automatic hyperparameter safety clamping, eliminating unnecessary `try` expressions.
- **Ergonomic Probability & Prediction APIs (`SwiftML`, `SwiftNLP`)**:
  - `ProbabilityMatrix`: Added 2D matrix subscripts `probs[row, col]`, row indexing `probs[row]`, and `probs.probabilities` array access.
  - `NaiveBayesClassifier`: Added direct single-instance prediction `predict(instance: [Double]) async throws -> Int`.
  - `MultinomialNaiveBayes` & `ComplementNaiveBayes`: Maintained first-class `Codable` struct status alongside concurrent actors for JSON persistence pipelines.
- **7-Domain Scientific Accuracy Scorecard (`Benchmarks`)**: Complete verification matrix across Tabular ML, LAPACK OLS, Clustering, NLP, AutoML, and Time Series validating 100% parity with Python scikit-learn and statsmodels.

---

## 🚀 What's New in v3.8.1

- **Tabular Column-Level Text & Lexical Profiler (G-018):** Native `DataFrame.profileTextColumn(_:)` computing 20 lexical metrics (TTR, hapax legomena, Shannon entropy, top-K terms) with multi-language stopword pruning (English, Ukrainian, German, French, Spanish).
- **Zero-Compromise Precision Optimizations (IEEE 754 64-bit Double Precision):**
  - **Direct In-Memory SQLite Ingestion:** Swift 6 actor-isolated persistent handle replacing disk roundtrips: **0.032 ms** (⚡ **3.28× faster than Pandas** 0.105 ms, 63× less RAM).
  - **Vectorized Naive Bayes Classifier:** Flat 1D memory buffers with Apple Accelerate `vDSP_dotprD` SIMD dot-products and Log-Sum-Exp trick: **0.026 ms** (⚡ **14.9× faster than Scikit-Learn** 0.388 ms).
  - **Holt-Winters Phase Shift Correction & 3D Nelder-Mead Optimization:** Seasonal index alignment and simplex parameter optimization achieving **$R^2 = 0.997$** and **RMSE = 0.350** (exact match with Statsmodels) in **0.645 ms** (⚡ **224× faster than Statsmodels**).
  - **Combinatorial TreeSHAP:** Precomputed 64×64 combinatorial weight LUT and zero-allocation in-place path backtracking over tree nodes: **0.103 ms** (11 MB vs 691 MB).
  - **Single-Pass Sparse TF-IDF Vectorizer:** Fast character-level tokenizer with single-pass sparse accumulation: **0.388 ms** (22 MB vs 691 MB).
  - **Hardware-Routed LinearSVC:** Automatic CPU/GPU router (`cells < 50_000 ? .cpu : .gpu`) and flat contiguous gradient buffers: **0.402 ms** (37 MB vs 668 MB).
  - **KMeans Clustering:** SIMD distance cache with floating-point underflow clamp: **11.192 ms** (⚡ **1.07× faster than Scikit-Learn** 11.993 ms).
  - **Zero-Allocation Time Series Decomposition:** Zero-allocation Kahan summation ($< 10^{-16}$ error): **0.088 ms** (⚡ **1.14× faster than Statsmodels** 0.100 ms).

---

## 🚀 What's New in v3.8.0

- **SwiftAgent Omni-Module Architecture (`SwiftSciToolbox`):** Unified high-level tool suite providing ready-to-use agent tools bridging all core scientific modules: DataFrame profiling and transformation (`SwiftDataFrame`), vectorized descriptive statistics and correlation (`SwiftStats`), automated model training and inference (`SwiftML`), time series forecasting (`SwiftForecast`), hyperparameter cross-validation (`SwiftOptimize`), asynchronous database query execution (`SwiftDatabase`), computer vision classification (`SwiftVision`), natural language tokenization and metrics (`SwiftNLP`), and feature attribution (`SwiftExplain`).
- **Typed Structured Tool Calling Protocol (`AgentToolV2`):** Type-safe tool specification protocol requiring explicit JSON parameter validation schemas (`JSONSchema`) and asynchronous execution handlers with structured dictionary inputs.
- **Real-Time Asynchronous Event Streaming (`ReActAgent.stream`):** Asynchronous stream interface yielding fine-grained execution events (`.thoughtDelta`, `.toolCallScheduled`, `.toolExecutionCompleted`, `.answerDelta`, and `.completed`) as reasoning evolves, enabling real-time terminal progress and interactive UI rendering.
- **Multi-Tier Agent Memory System (`AgentMemory`):** Modular conversation and episodic memory with short-term `WorkingMemory`, capacity-capped `SlidingWindowMemory`, and SIMD vDSP-accelerated `SemanticVectorMemory` for vector cosine similarity retrieval.
- **Native SwiftUI Reactive Dialogue Controller (`AgentDialogueController`):** Swift 6 `@Observable` controller providing end-to-end management of streaming dialogue history, cooperative task cancellation, error handling, and reactive state publication for SwiftUI frontends.
- **100.00% DocC Coverage & High-Coverage Test Suite:** Comprehensive documentation across all public symbols and extensive test coverage (96.53%+ overall patch coverage).

---

## 🚀 What's New in v3.7.0

- **GBDT CoreML Model Serialization (`SwiftML`):** Full `CoreMLExportable` conformance for `GradientBoostedTreesRegressor`, enabling direct export of trained gradient boosted tree ensembles to Apple Core ML specifications (`.mlmodel` / `.mlpackage`) with leaf shrinkage (`learningRate`), base initial prediction, and split threshold evaluation for sub-millisecond on-device inference.
- **Statistical Data Drift Detection (`SwiftStats`):** Added 1D Wasserstein distance (`Stats.wassersteinDistance(_:_:)`) via sorted cumulative empirical distributions and Population Stability Index (`Stats.populationStabilityIndex(reference:actual:numBins:)`) with quantile binning and Laplace smoothing.
- **Automated Modality Inference & Corpus Profiling (`SwiftDataFrame` & `SwiftNLP`):** Added `DataFrame.inferModality()` identifying dataset archetypes (`.tabularNumeric`, `.tabularMixed`, `.pureTextNLP`, `.timeSeries`) and `CorpusLexicalProfiler` computing Type-Token Ratio (TTR), Hapax Legomena, and empirical Shannon entropy.
- **Target Leakage Detection (`SwiftOptimize`):** Added `TargetLeakageDetector` auditing feature spaces for high Pearson correlation ($|r| \ge 0.98$), monotonic Spearman rank alignment, and row-index correlation prior to model training.
- **100% Rich DocC Documentation Quality Overhaul:** Eliminated all 1,239 `<#description#>` and `<#error description#>` placeholders across 117 files, replacing them with mathematically exact parameter, error, and return documentation, backed by zero-placeholder CI enforcement.

---

## 🚀 What's New in v3.6.1

- **Expanded WordNet Taxonomy (`SwiftNLP`):** Enriched default vocabulary from 6 synsets to a curated taxonomy of ~120 core synsets spanning organisms, human professions, artifacts, computing, sciences, mathematics, verbs, and qualitative adjectives with full hypernym/hyponym graph connectivity.
- **Princeton WordNet Data Ingestion Engine (`SwiftNLP`):** Native loader and parser (`WordNet.load(fromDataFile:pos:)`, `WordNet.load(fromDirectory:)`, `WordNet.parsePrincetonData`) capable of reading official Princeton WordNet database distribution files (`dict/data.noun`, `dict/data.verb`).
- **Database Driver Health Check & Connectivity Verification (`SwiftDatabase`):** Added async `ping() -> Bool` across all database drivers (`SQLiteConnection`, `PostgreSQLConnection`, `MySQLConnection`).
- **Symmetric Endpoint Initializers (`SwiftDatabase`):** Added convenience `init(host:port:user:password:database:sslMode:)` to `MySQLConnection` and `PostgreSQLConnection`.
- **Pure-Swift Wire Protocol Documentation Alignment:** Clarified full native implementations of PostgreSQL v3.0 (with SCRAM-SHA-256 and TLS) and MySQL Client/Server Protocol 4.1+ (with TLS) in DocC catalogs.

---

## 🚀 What's New in v3.6.0

- **HNSW Approximate Nearest Neighbor Graph Index (`SwiftCluster`):** Sub-millisecond $O(\log N)$ approximate nearest neighbor search over 100k+ high-dimensional embeddings with vDSP cosine/L2 acceleration.
- **256-Bin Histogram GBDT (`SwiftML`):** Tabular gradient boosting (`HistGradientBoostingClassifier` / `HistGradientBoostingRegressor`) with 256 discrete bins and $O(\text{numBins})$ split evaluations.
- **Concurrent TaskGroup Cross-Validation & AutoML (`SwiftOptimize`):** Bounded concurrent fold evaluations utilizing all CPU cores with cooperative task cancellation.
- **EarlyStopping Callbacks (`SwiftML`):** Training callbacks with `patience`, `minDelta`, and parameter restoration for MLP and GBDT.
- **CSR / CSC Sparse Matrices (`SwiftPreprocessing`):** Compressed sparse matrix storage accelerated with Apple Accelerate Sparse BLAS.
- **Parallel AutoARIMA Order Search (`SwiftForecast`):** Concurrent grid search for $(p,d,q) \times (P,D,Q)_s$ parameters.
- **Metal MSL SIMD-Group Quantization Kernels (`SwiftLLM`):** Custom GPU kernels for 4-bit/8-bit dequantization and GEMM using `simdgroup_matrix`.
- **Multi-Agent Orchestrator & Async Stream Bus (`SwiftAgent`):** `MultiAgentOrchestrator` and `AgentMessageBus` for multi-agent collaborative workflows.
- **SQLite 1024-Row Buffer Ingestion & SCRAM-SHA-256 (`SwiftDatabase`):** High-throughput buffered SQLite reading and RFC 5802/7677 compliant SCRAM-SHA-256 PostgreSQL authentication.
- **Enterprise Safety & Parity Hardening:** SVD Moore-Penrose pseudo-inverse in KalmanFilter, `svdFlip` PCA alignment, Welford two-pass variance, XSS-sanitization in Plotly export, Xoshiro256++ PRNG, byte-level BPE decoder, and 100% DocC API coverage with structured tags.

> For previous version notes (v3.5.2, v3.5.1, etc.), see [CHANGELOG.md](CHANGELOG.md) and [ROADMAP.md](ROADMAP.md).

---

## 📚 Documentation & Resources

- **API Reference**: [Interactive DocC Documentation](https://nodibell.github.io/SwiftSci/) (100% verified coverage)
- **The SwiftSci Book**: [Comprehensive Guides & Theory](Book/README.md)
- **Benchmarking Suite**: [PERFORMANCE.md](PERFORMANCE.md) & [ACCURACY.md](ACCURACY.md)
- **Release History**: [CHANGELOG.md](CHANGELOG.md)
- **Roadmap**: [ROADMAP.md](ROADMAP.md)

---

## 📄 License

SwiftSci is available under the MIT License. See [LICENSE](LICENSE) for details.
