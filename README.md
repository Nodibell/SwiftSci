# SwiftSci 3.3.0


**SwiftSci** is a native, high-performance, modular data analysis and machine learning library for Swift. It is built from the ground up to leverage Apple Silicon (M-series) unified memory architecture (UMA) and is fully compliant with Swift 6 strict concurrency requirements.

The package combines hardware-accelerated tensor computations on the Apple Silicon GPU via **MLX** with highly optimized CPU vector routines from the **Accelerate framework (vDSP / LAPACK / BLAS)** and Apple's native **NaturalLanguage** framework.

[![Swift Version](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/Nodibell/SwiftSci)
[![Platform Compatibility](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/Nodibell/SwiftSci)
[![codecov](https://codecov.io/gh/Nodibell/SwiftSci/graph/badge.svg)](https://codecov.io/gh/Nodibell/SwiftSci)
[![Documentation Coverage](https://img.shields.io/badge/DocC%20Coverage-100%25-brightgreen)](https://nodibell.github.io/SwiftSci/)

---

## 🌐 Platform Support & SPI Compatibility

SwiftSci is engineered for high-performance macOS execution across Apple Silicon ecosystems:

* **macOS 14+ (Apple Silicon M-Series)**: Supports all **14 core modules**, leveraging Accelerate (vDSP/LAPACK/BLAS) and MLX Metal GPU acceleration (`SwiftPreprocessing`, `SwiftML`, `SwiftCluster`, `SwiftLLM`, `SwiftExplain`, `SwiftVision`, `SwiftAgent`).

---

## What's New in 3.3.0

- **Core ML `Pipeline` Serialization (`SwiftML`):** Export composite end-to-end pipelines (`PipelineClassifier` / `PipelineRegressor`, fields 200/201 in Model.proto) chaining `StandardScaler` + `Encoders` + `RandomForestClassifier` or `MLPClassifier` in a single Core ML artifact.
- **Modern `.mlpackage` Directory Bundle Exporter (`SwiftML`):** Export models into modern `.mlpackage` directory bundles containing `Manifest.json` and nested model payloads via `CoreMLExporter.writeMLPackage` and `CoreMLExportable.writeMLPackage`.
- **In-Memory `VectorStore` Index (`SwiftCluster`):** High-throughput SIMD Accelerate-optimized vector index supporting Cosine Similarity, Dot Product, and Euclidean L2 distance for on-device semantic search and local RAG.
- **Batch Database Ingestion (`DataFrame.toSQL`, `SwiftDatabase`):** High-speed tabular bulk insertion into SQLite, PostgreSQL, and MySQL tables with `.append`, `.replace`, and `.failIfExists` modes.
- **Database TLS/SSL Security (`SSLMode`, `SwiftDatabase`):** Configurable `SSLMode` (`.disable`, `.prefer`, `.require`) with query string parsing (`?sslmode=require`, `?ssl=true`) for remote database connections.
- **Local Dense Text Embedding Engine (`SwiftNLP`):** Fast, offline dense text embedding generator (128-D/256-D L2-normalized vectors) operating 100% locally on Apple Silicon.
- **100% DocC API Coverage:** Maintained 100.00% public API documentation coverage (1,356 symbols) verified by automated CI.

See [CHANGELOG.md](CHANGELOG.md#330---2026-08-21).

---

## Core Modules

| Module                           | Description                                                                                                                                                                                                                                                                                                                                                          |                                    Docs                                    |
| :------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------: |
| **`SwiftDataFrame`**     | SIMD vectorised `filterFast` single-column evaluation (`vDSP_vcmprsD`), Accelerate `vDSP_vsortD` primitive Double sorting, Arrow zero-copy Feather (`FeatherReader`/`FeatherWriter`), deferred `LazyDataFrame` with filter pushdown, **`DataFrame.unique`** deduplication. |   [📖](https://nodibell.github.io/SwiftSci/documentation/swiftdataframe/)   |
| **`SwiftStats`**         | Vectorized descriptive statistics, SIMD vDSP sorting, Student-t/Chi-Square/F distributions, paired t-test, ANOVA powered by `Accelerate vDSP`.                                                                                                                                                                                                                      |     [📖](https://nodibell.github.io/SwiftSci/documentation/swiftstats/)     |
| **`SwiftPreprocessing`** | Feature scaling, categorical encoding (`OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`, `FrequencyEncoder`), imputation (`Imputer`, `KNNImputer`), `Pipeline`, `ColumnTransformer`, `HardwareRouter`, **Tier B Sendable Structs**.                                                                                                                               | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftpreprocessing/) |
| **`SwiftML`**            | Linear/Logistic Regression (LAPACK OLS `dgels_`), Decision Trees, Random Forests, GBDTs, **`LinearSVC`**, **`MLPClassifier`** & **`MLPRegressor`**, **Binary Core ML Exporter (`.mlmodel`)** (Linear, Trees, Forests, MLPs, Scalers), binary **ONNX exporter**, **`SwiftMLError`**. |      [📖](https://nodibell.github.io/SwiftSci/documentation/swiftml/)      |
| **`SwiftCluster`**       | Halko (2011) $O(MNk)$ `RandomizedSVD` for fast `PCA` (`svdSolver: .randomized`), divide-and-conquer SVD (`dgesdd_`), DBSCAN, `IsolationForest`, `LocalOutlierFactor`, `KMeans` with parallel `concurrentPerform` centroid assignment. |    [📖](https://nodibell.github.io/SwiftSci/documentation/swiftcluster/)    |
| **`SwiftOptimize`**      | `KFold`, `StratifiedKFold`, `TimeSeriesSplit` cross-validation, ROC-AUC, PR-AUC, MCC, `GridSearchCV`, generalized `RandomizedSearchCV` (`searchGeneric`).                                                                                                                                                                                                      |   [📖](https://nodibell.github.io/SwiftSci/documentation/swiftoptimize/)   |
| **`SwiftForecast`**      | 1D FIR moving average via `vDSP_convD`, ETS State Space model (`autoFit` AICc selection), Prophet-style `PiecewiseTrendDecomposition`, Exponential Smoothing, ARIMA, SARIMA, GARCH, Kalman filter, **`KoopmanOperator`** EDMD. |   [📖](https://nodibell.github.io/SwiftSci/documentation/swiftforecast/)   |
| **`SwiftNLP`**           | NLTK-equivalent engine: `AppleWordTokenizer`, `SentenceTokenizer`, `RegexTokenizer`, `PorterStemmer`, `POSTagger`, `AppleLemmaTagger`, `VADERSentimentAnalyzer`, `NGramTokenizer`, `HashingVectorizer`, actor-based **`NaiveBayesClassifier`** & **`ComplementNaiveBayesClassifier`** (`Codable`). |      [📖](https://nodibell.github.io/SwiftSci/documentation/swiftnlp/)      |
| **`SwiftExplain`**       | Black-box explainability via parallelized `KernelSHAP`, model-aware `TreeSHAP`, `PartialDependencePlot`, `PermutationImportance`, `TextExplainer`.                                                                                                                                                                                                          |    [📖](https://nodibell.github.io/SwiftSci/documentation/swiftexplain/)    |
| **`SwiftLLM`**           | `MLX.compile` forward pass caching per sequence-length bucket (16, 32, 64, 128, 256), `KVCache` Key-Value tensor cache, and `generateStream` streaming output. |      [📖](https://nodibell.github.io/SwiftSci/documentation/swiftllm/)      |
| **`SwiftVisualization`** | Native SwiftUI `Canvas` charting (`SwiftSciChartView` for line, bar, heatmap) + Plotly HTML chart exporters. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftvisualization/) |
| **`SwiftVision`**        | Computer vision & neural inference: **Real YOLOv8n Object Detection** (`CSPDarknet` + `PANet` + Decoupled Head + DFL), **`ONNXWeightReader`** Protobuf binary weight parser, `YOLOPreprocessor` (640x640 letterbox), **Deep Convolutional U-Net** segmentation (`UNetArchitecture` on MLX), CNN feature extraction. |    [📖](https://nodibell.github.io/SwiftSci/documentation/swiftvision/)    |
| **`SwiftDatabase`**      | Native SQLite C-driver connector, **native PostgreSQL (v3.0 wire protocol)**, and **native MySQL (Client/Server protocol)** drivers for zero-copy DataFrame ingestion via `DataFrame.fromSQL`. |   [📖](https://nodibell.github.io/SwiftSci/documentation/swiftdatabase/)   |
| **`SwiftAgent`**         | Structured DSL command parser (`filter`, `sample`, `select`, `head`, `tail`, `rename`, `dropnulls`, `fillnulls`, `groupby`) & RAG Context Summary Generator for local LLMs.                                                                                                                                                                                                                                                                                        |     [📖](https://nodibell.github.io/SwiftSci/documentation/swiftagent/)     |

---

## 📊 Complete Performance Comparison (SwiftSci 3.2.0 vs Python)

Official comparative benchmark suite results comparing **SwiftSci 3.2.0** (Release Build `-c release`) against Python data science libraries (**NumPy**, **Pandas**, **Scikit-Learn**, **Statsmodels**, **SHAP**, **PyTorch**, **Ultralytics**) on Apple Silicon (M-series / macOS 15 arm64).

Values are median times from the latest benchmark run. `n/a` means that the Python suite did not include an equivalent benchmark.

### 👁️ 1. Computer Vision & Neural Inference

| Benchmark Scenario                              | SwiftSci 3.2.0 (Swift) |       Python Baseline       |    Swift Speedup    |  Winner  |
| :---------------------------------------------- | :------------------: | :-------------------------: | :-----------------: | :------: |
| **YOLOv8Detector Detect** (640×640, real GPU) | **12.317 ms** (~81 FPS) | n/a | n/a | — |
| **YOLOPreprocessor Letterbox** (1920×1080 → 640×640) | **0.414 ms** | n/a | n/a | — |
| **UNetSegmentation Predict** (128×128 image) | **0.107 ms** | n/a | n/a | — |

### 📈 2. Forecasting

| Benchmark Scenario                              | SwiftSci 3.2.0 (Swift) |       Python Baseline       |    Swift Speedup    |  Winner  |
| :---------------------------------------------- | :------------------: | :-------------------------: | :-----------------: | :------: |
| **ARIMA(1,1,1) Fit** (50k pts) | **2.463 ms** | 212.621 ms (*Statsmodels*) | ⚡**86.34×** | 🟢 Swift |
| **ARIMA(1,1,1) Forecast** (horizon=24) | **2.566 ms** | 213.709 ms (*Statsmodels*) | ⚡**83.27×** | 🟢 Swift |
| **Holt-Winters Fit** (50k pts, period=12) | **6.451 ms** | 144.752 ms (*Statsmodels*) | ⚡**22.44×** | 🟢 Swift |

### 🤖 3. Machine Learning & Clustering

| Benchmark Scenario                           | SwiftSci 3.2.0 (Swift) |       Python Baseline       |   Swift Speedup   |  Winner  |
| :------------------------------------------- | :--------------------: | :-------------------------: | :---------------: | :------: |
| **LinearSVC Fit** (1k×4, Metal GPU) | **0.463 ms** | n/a | n/a | — |
| **RandomForest Fit** (1k×4, 50 trees) | **3.838 ms** | 25.300 ms (*Scikit-Learn*) | ⚡**6.59×** | 🟢 Swift |
| **GBDT Regressor Fit** (1k×4, 50 estimators) | **8.397 ms** | 32.366 ms (*Scikit-Learn*) | ⚡**3.85×** | 🟢 Swift |
| **KMeans Fit** (10k×4, 3 clusters) | **20.062 ms** | 11.993 ms (*Scikit-Learn*) | 0.60× | 🔴 Python |
| **PCA SVD Fit** (1k×100 → 10 comps) | **1.014 ms** | 0.732 ms (*Scikit-Learn*) | 0.72× | 🔴 Python |

### 📝 4. Natural Language & Explainability

| Benchmark Scenario                            | SwiftSci 3.2.0 (Swift) |    Python Baseline    |   Swift Speedup   |  Winner  |
| :-------------------------------------------- | :------------------: | :-------------------: | :----------------: | :------: |
| **KernelSHAP Explain** (5 features, 100 coalitions) | **0.168 ms** | 0.413 ms (*SHAP*) | ⚡**2.45×** | 🟢 Swift |
| **LLM Forward Pass** (seqLen=64) | **0.437 ms** | 0.531 ms (*PyTorch*) | ⚡**1.21×** | 🟢 Swift |
| **LLM Generate** (10 tokens) | **4.936 ms** | 3.609 ms (*PyTorch*) | 0.73× | 🔴 Python |

### 📊 5. Core Data Engines & Vector Stats

| Benchmark Scenario                            | SwiftSci 3.2.0 (Swift) |    Python Baseline    |   Swift Speedup   |  Winner  |
| :-------------------------------------------- | :------------------: | :-------------------: | :----------------: | :------: |
| **Mean Reduction** (vDSP 1M elements) | **0.082 ms** | 0.121 ms (*NumPy*) | ⚡**1.48×** | 🟢 Swift |
| **StdDev Reduction** (vDSP 1M elements) | **0.275 ms** | 0.533 ms (*NumPy*) | ⚡**1.94×** | 🟢 Swift |
| **Variance Reduction** (vDSP 1M elements) | **0.282 ms** | 0.517 ms (*NumPy*) | ⚡**1.84×** | 🟢 Swift |
| **Pearson Correlation** (500k pairs) | **0.812 ms** | 1.193 ms (*NumPy*) | ⚡**1.47×** | 🟢 Swift |
| **Kalman Filter 1D** (10k observations) | **57.970 ms** | 85.788 ms (*NumPy*) | ⚡**1.48×** | 🟢 Swift |
| **CSV Read** (100k rows) | **15.465 ms** | 19.413 ms (*Pandas*) | ⚡**1.26×** | 🟢 Swift |
| **CSV Stream + GroupBy** (100k rows) | **22.830 ms** | 27.603 ms (*Pandas*) | ⚡**1.21×** | 🟢 Swift |

> ℹ️ **Transparent Reporting**: For the complete 36-benchmark matrix, see [PERFORMANCE.md](PERFORMANCE.md).

---

## 🛠 Architectural Highlights & Optimizations

### 1. Data-Oriented Design (DOD) Tree Ensembles

Traditional object-oriented trees suffer from ARC overhead and cache misses. `SwiftSci` stores trees as contiguous flat arrays of `FlatTreeNode` structures with pre-sorted feature matrix split indexes, eliminating pointer chasing and maximizing CPU L1/L2 cache hit rates.

### 2. Intelligent Hardware Routing

We implement a flexible compute device routing policy (`requestedDevice` / `resolvedDevice`):

* Branch-heavy algorithms (Decision Trees, Random Forests, spatial DBSCAN search) execute on CPU vector cores.
* Tensor operations (Linear/Logistic Regression, Multi-Layer Perceptrons) leverage Apple Silicon GPU execution via `MLXArray` lazy evaluation and Accelerate `BLAS` (`cblas_dgemm`).

### 3. Native NaturalLanguage Integration & SwiftNLP

`SwiftNLP` bridges NLTK-equivalent algorithms (`PorterStemmer`, `VADERSentimentAnalyzer`, `NaiveBayesClassifier`) with native Apple OS frameworks (`NLTagger`, `NLLanguageRecognizer`, `NLEmbedding`), ensuring high accuracy and zero external binary dependencies.

---

## 💻 Quick Start

```swift
import Foundation
import SwiftDataFrame
import SwiftML
import SwiftNLP

// 1. Create columnar DataFrame
let xCol = TypedColumn<Double>(name: "x", values: [1.0, 2.0, 3.0, 4.0, 5.0])
let yCol = TypedColumn<Double>(name: "y", values: [2.0, 4.0, 6.0, 8.0, 10.0])
let df = try DataFrame(columns: [xCol, yCol])

// 2. Analytical OLS Linear Regression fit
let regressor = LinearRegression()
let X = try df.toFeatureMatrix(columnNames: ["x"])
let y = try df.toTargetVector(columnName: "y")
try regressor.fit(features: X, target: y)

// 3. Native Tokenization & Sentiment Analysis
let vader = VADERSentimentAnalyzer()
let score = vader.polarityScores(text: "SwiftSci 3.2.0 is super fast!")
print("Sentiment compound score:", score.compound)
```
