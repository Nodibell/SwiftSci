# SwiftSci 3.13.0

**SwiftSci** is a native, high-performance, modular scientific computing and machine learning library for Swift. Built from the ground up for Apple Silicon (M-series) Unified Memory Architecture (UMA), SwiftSci is fully compliant with Swift 6 strict concurrency requirements.

It seamlessly blends hardware-accelerated tensor operations on the Apple Silicon GPU via **MLX** with ultra-optimized CPU vector routines from the **Accelerate framework (vDSP / LAPACK / BLAS)** and Apple's native **NaturalLanguage** framework.

[![Swift Version](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/Nodibell/SwiftSci)
[![Platform Compatibility](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/Nodibell/SwiftSci)
[![codecov](https://codecov.io/gh/Nodibell/SwiftSci/graph/badge.svg)](https://codecov.io/gh/Nodibell/SwiftSci)
[![Documentation Coverage](https://img.shields.io/badge/API%20Doc%20Coverage-100%25-brightgreen)](https://nodibell.github.io/SwiftSci/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![XERJ Corpus Hub contributor](https://hub.xerj.org/contributors/Nodibell/badge.svg)](https://hub.xerj.org/contributors/Nodibell)

---

## ⚡ Key Architectural Advantages

- 🚀 **Apple Silicon Native (UMA)**: Zero-copy tensor and columnar memory sharing between CPU and Metal GPU via MLX.
- 🛡️ **Swift 6 Strict Concurrency**: Designed for Swift 6 strict concurrency, with Sendable-safe APIs and actor isolation where required.
- 📦 **15 Integrated Modules**: Full end-to-end data stack from Apache Parquet & Arrow DataFrames to GBDT, Time Series ARIMA, XAI (TreeSHAP/LIME), Local LLMs, Computer Vision, Autonomous ReAct Agents, and persisted ML Workflows.
- 📉 **Radical Memory Efficiency**: Consumes **10× to 70× less RAM** than Python runtime stacks (Pandas, NumPy, Scikit-Learn, PyTorch).
- 🧩 **Zero-Boilerplate Ingestion**: Pure-Swift readers for Apache Parquet, Apache Arrow Feather, NumPy `.npy`/`.npz`, and automatic SQLite schema discovery.

---

## 📦 Installation

Add SwiftSci to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/Nodibell/SwiftSci.git", from: "3.13.0")
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

> **Platform & Toolchain Requirements:** macOS 14+ (Apple Silicon M-Series), Swift 6.3+ (Xcode 26.3+, required by `mlx-swift` 0.33.0+).

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
let X = try df.toFeatureMatrix(["feature"])
let y = try df.toTargetVector("target")
try await regressor.fit(features: X, targets: y)

// 3. Fast Categorical OneHotEncoder (5× faster than Scikit-Learn)
let ohe = OneHotEncoder()
try ohe.fit([["engineering"], ["finance"], ["engineering"]])
let encoded = try ohe.transform([["engineering"], ["finance"]])

// 4. Black-Box Model Explainability (TreeSHAP & LIME)
let lime = LIMEExplainer(kernelWidth: 0.75, regularization: 0.01)
let explanation = await lime.explain(model: { $0.reduce(0.0, +) }, instance: [1.0, 2.0, 3.0], numSamples: 300)

// 5. Native Sentiment Analysis
let vader = VADERSentimentAnalyzer()
let score = vader.polarityScores(text: "SwiftSci 3.13.0 is incredibly fast, memory efficient, and robust!")
print("Sentiment compound score:", score.compound)
```

---

For a complete CSV-to-predictions workflow with training-only preprocessing, PCA,
classification, saved models and prediction on unlabeled data, run the
[numerical workflow example](Examples/NumericalWorkflow/README.md).

The optional `SwiftWorkflow` module composes these stages through
`PCAClassificationWorkflow`. It saves the fitted parameters and feature schema
in one versioned JSON artifact. The 14 core modules remain independently usable.

## 🧩 The 15 Integrated Modules

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
| **`SwiftDatabase`** | Zero-copy SQL ingestion/export: SQLite with **1024-row column buffer ingestion**, auto-discovery, and C-API read-only query mode; native PostgreSQL (`SCRAM-SHA-256` / `MD5`) and MySQL (`mysql_native_password` SHA-1; unsupported plugins fail closed) wire connectors with fail-closed `SSLMode.require` protection. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftdatabase/) |
| **`SwiftAgent`** | Omni-module autonomous agents: **`SwiftSciToolbox`** scientific tools, typed **`AgentToolV2`** with JSON schemas, **`ReActAgent`** with real-time async event streaming, **`SemanticVectorMemory`** (HNSW graph recall), and native SwiftUI **`AgentDialogueController`** (`@Observable`). | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftagent/) |
| **`SwiftVisualization`** | Native SwiftUI `Canvas` interactive charts (`SwiftSciChartView`), XSS-sanitized Plotly HTML export, and terminal ASCII/Braille graphs. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftvisualization/) |
| **`SwiftWorkflow`** | End-to-end persisted preprocessing, PCA, and classification pipelines (`PCAClassificationWorkflow`) with versioned JSON artifact serialization and strict feature schema validation. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftworkflow/) |

---

## 🧪 Two-Tier Verification & Testing Policy

Because GitHub-hosted `macos-latest` runners operate in virtualized environments without a physical Apple Silicon Metal GPU device required by `MLX`, SwiftSci employs a documented two-tier verification workflow:

| Verification Tier | Environment | Scope & Responsibilities |
| :--- | :--- | :--- |
| **Tier 1: Continuous Integration** | GitHub Actions (`macos-latest`) | Full Debug & Release compilation of all 15 modules, DocC symbol-graph generation & 100% public API declaration verification (`scripts/verify_doc_coverage.py`), and pure-CPU test suites (`SwiftDataFrame`, `SwiftStats`, `SwiftForecast`, `SwiftNLP`, `SwiftDatabase`, `SwiftAgent`, `SwiftVisualization`). |
| **Tier 2: Hardware Release Gate** | Local Apple Silicon (M-Series) | Complete `swift test` execution across all 15 modules (including `SwiftPreprocessing`, `SwiftML`, `SwiftCluster`, `SwiftExplain`, `SwiftOptimize`, `SwiftLLM`, `SwiftVision`, `SwiftWorkflow`), Metal GPU kernels, and cross-project API compatibility verification (`CookbookVerification`, `Saura`). |

---

## 📊 Performance Comparison: SwiftSci 3.13.0 vs 3.12.0 vs 3.11.0 vs Python

The following benchmark metrics reflect the standardized, certified dual-language automated test harness (`Benchmarks/Runs/standard-3.12.0` and `Benchmarks/Runs/migration-3.12.0`) run on Apple Silicon (M4 Pro, IEEE 754 64-bit Double Precision, Swift 6 Release vs Python with Pandas, NumPy, Scikit-Learn, SciPy, and Statsmodels) under **Benchmark Methodology v2**. Because **SwiftSci 3.13.0** focuses on upstream `mlx-swift 0.33.0` alignment, native packed quantized LLM loading, and `SwiftLLM`/`SwiftNLP` hardening without changes to CPU, DataFrame, preprocessing, statistical, or classical ML paths, tabular and ML benchmarks carry forward the audited `3.12.0` baseline:

### ⚡ Side-by-Side: SwiftSci 3.13.0 vs 3.12.0 vs 3.11.0 (100k Rows / ML Fixtures)

| Benchmark Scenario | SwiftSci 3.11.0 | SwiftSci 3.12.0 | SwiftSci 3.13.0 | Python Baseline | 3.13 vs 3.11 | 3.13 vs Python |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CSV Read** (100k rows × 5 cols) | `3.76 ms` | `2.20 ms` | **`2.20 ms`** | `9.92 ms` (*Pandas*) | ⚡ **1.70× faster** | **SwiftSci 4.50×** |
| **CSV Stream Read** (100k rows, 10k chunks) | `46.80 ms` | `2.21 ms` | **`2.21 ms`** | `12.82 ms` (*Pandas*) | ⚡ **21.2× faster** | **SwiftSci 5.81×** |
| **CSV Stream Filter** (100k rows, 10k chunks) | `—` | `2.58 ms` | **`2.58 ms`** | `14.15 ms` (*Pandas*) | Added in 3.12 | **SwiftSci 5.49×** |
| **CSV Stream GroupBy** (100k rows, 10k chunks)| `—` | `2.94 ms` | **`2.94 ms`** | `24.26 ms` (*Pandas*) | Added in 3.12 | **SwiftSci 8.27×** |
| **Sort Double Column** (100k rows) | `2.75 ms` | `2.14 ms` | **`2.14 ms`** | `2.90 ms` (*NumPy*) | ⚡ **1.29× faster** | **SwiftSci 1.35×** |
| **MinMaxScaler (Fit + Transform)** (100k) | `0.73 ms` | `0.74 ms` | **`0.74 ms`** | `0.75 ms` (*Scikit-Learn*) | Parity | **SwiftSci 1.01×** |
| **StandardScaler (Fit + Transform)** (100k) | `1.97 ms` | `1.97 ms` | **`1.97 ms`** | `0.88 ms` (*Scikit-Learn*) | Parity | Python 2.22× |
| **Direct Prepared Matrix Pipeline** (100k) | `2.70 ms` | `2.70 ms` | **`2.70 ms`** | `1.64 ms` (*Scipy/Sklearn*) | Parity | Consuming zero-copy |
| **Target Vector Extraction** (100k) | `0.051 ms` | `0.057 ms` | **`0.057 ms`** | `0.045 ms` (*NumPy*) | Parity | Parity (1.27×) |
| **Flat Feature Matrix Extraction** (100k) | `0.113 ms` | `0.112 ms` | **`0.112 ms`** | `0.659 ms` (*NumPy*) | Parity | **SwiftSci 5.88×** |
| **Row Filtering (Boolean mask)** (100k) | `0.23 ms` | `0.23 ms` | **`0.23 ms`** | `0.64 ms` (*Pandas*) | Parity | **SwiftSci 2.82×** |
| **GroupBy + Aggregation** (100k rows) | `0.37 ms` | `0.38 ms` | **`0.38 ms`** | `0.51 ms` (*Pandas*) | Parity | **SwiftSci 1.34×** |
| **GroupBy Multi-Output (Sum + Mean)** (100k) | `—` | `0.49 ms` | **`0.49 ms`** | `1.60 ms` (*Pandas*) | Added in 3.12 | **SwiftSci 3.29×** |
| **Row-wise Horizontal Sum** (100k rows) | `—` | `3.85 ms` | **`3.85 ms`** | `18.96 ms` (*Pandas*) | Added in 3.12 | **SwiftSci 4.92×** |
| **DataFrame SIMD Hash Join** (100k rows) | `8.47 ms` | `8.47 ms` | **`8.47 ms`** | `0.46 ms` (*Pandas*) | Parity | Vectorized SIMD probe |
| **RandomForest Fit** (1k×4, 50 trees) | `2.21 ms` | `2.21 ms` | **`2.21 ms`** | `31.18 ms` (*Scikit-Learn*) | Parity | **SwiftSci 14.11×** |
| **GBDT Regressor Fit** (1k×4, 50 est) | `4.68 ms` | `4.68 ms` | **`4.68 ms`** | `32.35 ms` (*Scikit-Learn*) | Parity | **SwiftSci 6.91×** |
| **OLS Linear Fit + Predict** (1k × 3) | `0.080 ms` | `0.080 ms` | **`0.080 ms`** | `0.85 ms` (*Scikit-Learn*) | Parity | **SwiftSci 10.63×** |
| **Peak Resident Memory (CSV 100k RSS)** | `48.7 MiB` | `32.2 MiB` | **`32.2 MiB`** | `211.6 MiB` (*Pandas*) | ⚡ **1.51× less RAM** | **6.57× less RAM** |

### 📊 Full Workload Matrix (SwiftSci 3.13.0 Baseline vs Reference Stack)

| Domain / Scenario | SwiftSci 3.13.0 | Python Reference Stack | Relative Performance | Scope / Implementation |
| :--- | :---: | :---: | :---: | :--- |
| **Holt-Winters Fit** (50k pts, s=12) | **`0.601 ms`** | `3,299.06 ms` (*Statsmodels*) | **SwiftSci >5000×** | Native Nelder-Mead simplex optimizer |
| **ARIMA(1,1,1) Fit** (50k pts) | **`0.041 ms`** | `392.75 ms` (*Statsmodels Kalman MLE*) | **SwiftSci >9000×** | Vectorized Hannan-Rissanen conditional OLS solver (see methodology note) |
| **NaiveBayes Fit** (1k×100, 3 classes) | **`0.047 ms`** | `0.410 ms` (*Scikit-Learn*) | **SwiftSci 8.72×** | Accelerate `vDSP_dotprD` SIMD dot-product |
| **SQLite DataFrame Ingestion** | **`0.025 ms`** | `0.443 ms` (*Pandas*) | **SwiftSci 17.90×** | Persistent in-memory C-API (63× less RAM) |
| **RandomForest Fit** (1k×4, 50 trees, d=4, gini) | **`2.210 ms`** | `31.18 ms` (*Scikit-Learn*) | **SwiftSci 14.11×** | Data-Oriented Design (DOD) tree buffers |
| **GBDT Regressor Fit** (1k×4, 50 est.) | **`4.680 ms`** | `32.35 ms` (*Scikit-Learn*) | **SwiftSci 6.91×** | Contiguous gradient-boosted ensembles |
| **CSV Stream GroupBy** (100k rows, 10k chunks) | **`2.935 ms`** | `24.26 ms` (*Pandas*) | **SwiftSci 8.27×** | On-demand streaming chunks & compact grouping (11.8× less RAM) |
| **CSV Stream Read** (100k rows, 10k chunks) | **`2.206 ms`** | `12.82 ms` (*Pandas*) | **SwiftSci 5.81×** | Scoped record batch discovery & SIMD structural scan (21.2× vs 3.11.0) |
| **CSV Read** (100k rows × 5 cols) | **`2.203 ms`** | `9.92 ms` (*Pandas*) | **SwiftSci 4.50×** | Vectorized unquoted CSV scan & compact `Double` retention (1.70× vs 3.11.0) |
| **OneHotEncoder** (50k rows) | **`4.912 ms`** | `27.64 ms` (*Scikit-Learn*) | **SwiftSci 5.63×** | SIMD categorical bitmask transform |
| **Classification ROC-AUC** (50k preds) | **`2.115 ms`** | `2.08 ms` (*Scikit-Learn*) | **Parity 1.02×** | Single-pass sorted trapezoidal integration (1.22× faster than 3.11.0) |
| **KernelSHAP Explain** (100 coalitions) | **`0.189 ms`** | `0.434 ms` (*SHAP*) | **SwiftSci 2.30×** | Zero-division guarded coalitional sampling |
| **Two-Sample Welch T-Test** (100k samples) | **`0.284 ms`** | `0.632 ms` (*SciPy*) | **SwiftSci 2.22×** | Accelerate vDSP Welch's t-test (7.9× less RAM) |
| **Student / Paired / ANOVA** (100k samples) | **`0.272–0.402 ms`** | `0.540–0.811 ms` (*SciPy*) | **SwiftSci 2.02–2.38×** | Compensated blocked moments & Accelerate distributions |
| **Pearson Correlation** (100k / 500k pairs) | **`0.148 / 0.791 ms`** | `0.254 / 1.202 ms` (*NumPy*) | **SwiftSci 1.52–1.72×** | SIMD dot-product covariance |
| **Mean Reduction** (vDSP 1M doubles) | **`0.081 ms`** | `0.118 ms` (*NumPy*) | **SwiftSci 1.46×** | Apple Accelerate `vDSP_meanvD` |
| **SortBy Double Column** (100k rows) | **`2.140 ms`** | `2.896 ms` (*NumPy*) | **SwiftSci 1.35×** | Compact `Double` adaptive radix sort (1.29× faster than 3.11.0) |
| **LinearSVC Fit** (1k×4, 100 epochs, Metal GPU) | **`0.751 ms`** | `0.384 ms` (*Scikit-Learn*) | **Python 1.95×** | Apple Metal GPU kernel vs LibLinear C |
| **PCA SVD fitTransform** (1k×100 → 10 comps) | **`0.027 ms`** | `0.758 ms` (*Scikit-Learn*) | **SwiftSci 28.07×** | Accelerate LAPACK `dgesdd_` SVD spectrum |
| **KMeans Fit** (10k×4, 3 clusters, 50 iters) | **`14.20 ms`** | `6.94 ms` (*Scikit-Learn*) | **Python 2.05×** | Underflow-clamped SIMD distance vs Cython k-means |
| **DataFrame SIMD Hash Join** (100k rows) | **`8.47 ms`** | `0.456 ms` (*Pandas*) | **Python 18.57×** | Zero-boxing flat chained hash table + direct typed buffer gather (4.15× speedup from 35.2 ms) |
| **SwiftLLM Incremental Decode** (RoPE + KV-Cache) | **`0.125 ms`** | `1.450 ms` (*Full Forward*) | **SwiftSci 11.60×** | Architectural comparison: single-token incremental decode with cached K/V ($O(N)$ attention) vs full-sequence forward recomputation ($O(N^2)$); $\Delta \le 1.03 \times 10^{-7}$ |
| **Metal MSL gemv_q4_0 Kernel** (1024d) | **`0.015 ms`** | `0.045 ms` (*Float32 Dequant*) | **SwiftSci 3.00×** | Zero-copy packed Q4_0 evaluation |
| **AutoARIMA Zero-Variance Guard** (1000 pts) | **`< 0.01 ms`** | Divergent loop | **Instant Exit** | Zero-variance fast exit + bounded optimization iterations |

### 🎯 Validation & Correctness Scorecard

SwiftSci incorporates an automated validation scorecard confirming numerical parity against reference Python baselines (Scikit-Learn, Statsmodels, SciPy).

> **Methodology Note**: Validation distinguishes between **Algorithmic Correctness** (parity with reference math), **Predictive Quality** (metrics on benchmark datasets), and **Performance**.
> - **ARIMA(1,1,1)** uses fast Hannan-Rissanen conditional OLS parameter estimation; on the non-stationary benchmark horizon ($h=24$), SwiftSci achieves RMSE 10.218 vs Statsmodels' 22.158 (2.17× lower error; out-of-sample $R^2$ is omitted as uninformative for non-stationary series).
> - **NaiveBayes** matches Scikit-Learn with **100% exact numerical parity** ($\Delta \text{Acc} = 0.00\%$) on the synthetic token distribution benchmark.

```text
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │                    VALIDATION & CORRECTNESS SCORECARD                              │
  ├────────────────────────────────────────────────────────────────────────────────────┤
  │ [Forecast] Holt-Winters (h=24) : RMSE=0.350, MAE=0.281, MAPE=0.21%, R²=0.997       │
  │ [Forecast] ARIMA(1,1,1) (h=24) : RMSE=10.218, MAE=8.557, MAPE=5.87% (Statsmodels: 22.16) │
  │ [ML Reg]   GBDT (30 trees, d=4) : RMSE=0.421, MAE=0.344, R²=0.9879                 │
  │ [ML Cls]   RandomForest (30 tr.): Accuracy=98.50%, F1=0.986, ROC-AUC=0.999         │
  │ [NLP Cls]  NaiveBayes (3-class) : Parity ΔAcc=0.00% (Matches Scikit-Learn baseline)│
  └────────────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 What's New in v3.13.0

- **Upstream `mlx-swift 0.33.0` Alignment (`Package.swift`)**: Switched dependency from forked `mlx-swift` to upstream `https://github.com/ml-explore/mlx-swift.git` (`0.33.0+`), keeping strict source compatibility for all existing SwiftSci callers.
- **Native Packed Quantized LLM Execution (`SwiftLLM`)**:
  - `TransformerDecoder.loadWeights(_:)` auto-detects `<module>.scales`, `<module>.biases`, and bit-packed `UInt32` weights (including across sharded files) and routes through `loadQuantizedWeights(_:quantization:)` so 4-bit/8-bit checkpoints stay packed on GPU via `quantized_matmul` instead of expanding to Float16.
  - Added `AffineQuantizationConfig` (with `init?(configJSON:)` and automatic shape inference) and cross-shard split-tensor staging so shard-by-shard callers get bit-exact outputs even when `.weight`, `.scales`, and `.biases` cross shard boundaries.
- **Zero-Configuration Model Download & Load (`SwiftLLM`)**: Added `ModelDownloader`, `LLMConfig(huggingFaceConfigJSON:maxSeqLen:)`, and streaming sharded `TransformerDecoder.load(directory:maxSeqLen:)`.
- **Parser & Tokenizer Hardening (`SwiftLLM`, `SwiftNLP`)**:
  - Added `U8`, `U16`, `U32`, `U64`, `I16`, and `BOOL` to `SafeTensorsParser`, plus bounds checks against hostile headers, inverted offsets, negative/overflowing tensor shapes, and payload length mismatches.
  - Fixed `GGUFParser` 1-byte boolean metadata alignment and added range/overflow checks on arrays, strings, dimensions, and offsets.
  - Updated `BPETokenizer(llama3TokenizerJSON:)` to accept bare `ByteLevel` `post_processor` configurations from community MLX exports.
- **Thread-Safe Sentence Embeddings (`SwiftNLP`)**: Serialized `NLEmbedding` vector lookups and switched `AppleSentenceEmbeddingEngine.embedBatch(_:)` to sequential cancellable execution, eliminating CoreFoundation `SIGSEGV` crashes on batches exceeding 16 texts.

## 🚀 What's New in v3.12.0

- **Persisted ML Workflows & `SwiftWorkflow` Module (`SwiftWorkflow`, `Examples/NumericalWorkflow`)**: Introduced the 15th module (`PCAClassificationWorkflow`) and the `ScientificWorkflow` CLI, composing training-only imputation, standardization, PCA projection, and classification into a versioned JSON artifact with strict feature schema validation, configurable CSV training, and labeled evaluation of saved models (PR #98–#101).
- **Prepared Classification, PCA & Reusable CPU MLP (`SwiftML`, `SwiftCluster`, `SwiftPreprocessing`)**: End-to-end `PreparedNumericBatch` fitting, projection, and prediction for `PCA`, `LogisticRegression`, `LinearSVC`, `ClassificationPipeline`, and reusable standardized CPU `MLP` inference with provenance accounting and high-throughput tile preprocessing (PR #82, #83, #84, #89, #90, #91, #93).
- **Vectorized & Streaming CSV Overhaul + Compact `Double` Retention (`SwiftDataFrame`)**: SIMD structural scans for unquoted CSV records, scoped record batch discovery, on-demand streaming chunks, explicit per-column date formats, and zero-expansion compact `Double` retention across CSV decoding, gathering, nullable joins, filtering, and adaptive radix sorting (PR #85, #86, #87, #88).
- **WAPE Metric, Zero-Shot Classification & Deterministic Embeddings (`SwiftForecast`, `SwiftNLP`, `SwiftAgent`)**: Added SIMD-accelerated `TimeSeriesMetrics.evaluateWAPE` (G-029), `ZeroShotClassifier` (G-025), cross-process deterministic FNV-1a local embeddings, and producer-linked `ReActAgent` stream cancellation (PR #80, #95, #96).
- **`ProbabilityMatrix` Collection Contract Fix (`SwiftML`)**: Fixed `RandomAccessCollection` conformance in `ProbabilityMatrix` (`endIndex == rows.count`, strict bounds-checked `subscript`), eliminating duplicate row predictions on $0 < N < 6$ batches and out-of-bounds traps on empty batches; added `Equatable` conformance (PR #102).
- **Database Security & Read-Only Enforcement (`SwiftDatabase`, `SwiftAgent`)**:
  - `PostgreSQLConnection` and `MySQLConnection` now fail closed (`DatabaseError.notImplemented`) when `sslMode == .require` (rejecting connections rather than silently downgrading to plaintext TCP while native TLS negotiation is not yet implemented; `sslMode == .prefer` continues to connect over unencrypted TCP without TLS negotiation) (PR #102).
  - Implemented standard 20-byte `mysql_native_password` SHA-1 challenge-response authentication (`SHA1(password) XOR SHA1(scramble20 + SHA1(SHA1(password)))`), dynamic `HandshakeResponse41` capability flags (`CLIENT_CONNECT_WITH_DB`, `CLIENT_PLUGIN_AUTH`), and strict `OK_Packet` (`0x00`) verification with explicit fail-closed rejection (`DatabaseError.unsupportedAuth`) when the server requests other authentication plugins or sends `AuthSwitchRequest` (`0xFE`) / `AuthMoreData` (`0x01`) (PR #102).
  - Added engine-enforced `SQLiteConnection.executeReadOnlyQuery(_:)` with a dedicated `SQLITE_OPEN_READONLY` handle (`readOnlyDbBox`) for file-backed databases (isolated from read-write handles on the same connection), `PRAGMA query_only = ON`, single-statement check, and `sqlite3_stmt_readonly`, plus tokenized SQL keyword validation in `SafeDatabaseQueryTool` (resolving false-positive rejections on identifiers like `created_at` / `updated_at`) (PR #102).
- **Class-Aware YOLOv8 NMS & Validated HNSW Indexing (`SwiftVision`, `SwiftCluster`)**:
  - `YOLOv8Detector` now applies class-aware Non-Maximum Suppression via `NonMaximumSuppression.filter`, preserving co-located detections of distinct classes, and supports standard ONNX tensor wire fields (PR #80, #81, #102).
  - `HNSWIndex` enforces vector dimensionality invariants (`dimension: Int?`) across `add`, `search`, and internal distance functions (removing `min(a.count, b.count)`), and introduces validated throwing APIs (`addValidated`, `searchValidated`, `HNSWError`) (PR #102).
- **Statistical & Visualization Hardening (`SwiftStats`, `SwiftForecast`, `SwiftVisualization`)**:
  - Added `NaN`/`Inf` input guards to `Stats.spearmanCorrelation` and finite non-negative `epsilon` validation to `TimeSeriesMetrics` (`evaluateMAPE`, `evaluateSMAPE`, `evaluateWAPE`) (PR #102).
  - Fixed pairwise-complete index alignment in `plotCorrelationHeatmap` and tied-score grouping in `plotROCCurve` (PR #102).
- **Toolchain Alignment & 100% API Documentation (`Package.swift`, `DocC`, `CI`)**:
  - Aligned `Package.swift` to `swift-tools-version: 6.3` matching `mlx-swift` 0.32.3+ requirements (PR #102).
  - Achieved **100.00% (2,525/2,525)** public declaration documentation coverage across all 15 modules (including `SwiftWorkflow`), expanded multi-section DocC tutorials, exported multi-binary LCOV coverage, and enabled `SwiftForecastTests` in GitHub Actions CPU CI (PR #80, #92, #97, #102).

## 🚀 What's New in v3.11.0

- **SIMD-Accelerated Hash Join (`SwiftDataFrame`)**: Vectorized hash probing and contiguous column equality dispatch across typed buffers (`Int64`, `Int32`, `Double`, `Float`, `String`, `Bool`), achieving **4.15× speedup** on inner joins with zero existential boxing (PR #73).
- **Direct Prepared Matrix Preprocessing (`SwiftDataFrame`, `SwiftPreprocessing`, `SwiftML`)**: Zero-copy consuming matrix pipeline (`PreparedNumericMatrix` $\to$ `Imputer.transform(consuming:)` $\to$ `StandardScaler.transform(consuming:)` $\to$ `MLP.predict(features:)`), cutting heap allocations from 42 MiB to 16 MiB with 1.16–1.32× speedup (PR #71, PR #72).
- **Native Llama 3.2 & MLX 0.32.3 (`SwiftLLM`)**: End-to-end native execution of Llama 3.2 models with stabilized RoPE dynamic frequencies, strict context limits, and upstream Float16 SIMD backport (PR #49, PR #50, PR #60).
- **YOLOv8 Object Detection (`SwiftVision`)**: Real-time object detection architecture with bounding box NMS and $mAP@50$ / $mAP@50:95$ evaluation metrics (PR #54, PR #55).
- **Modular AutoML & Evaluation Harness (`SwiftOptimize`)**: Production-grade AutoML with honest per-fold leaderboards, soft deadline budgeting, `StratifiedKFold`, Gorodkin multiclass MCC, and post-selection winner refitting (PR #54, PR #56, PR #57, PR #58).
- **Bounded Inference & Session Retention (`SwiftML`, `WiredMemoryManager`)**: Production memory admission policy bounding memory from source loading through consumption, scoped Core ML session retention, and mixed CPU/GPU scheduling (PR #52, PR #62, PR #65, PR #67, PR #68, PR #70).
- **Comprehensive Datetime & Arrow Ingestion (`SwiftDataFrame`)**: High-precision ISO 8601 millisecond datetime parsing, explicit column casting (`castColumn`), and optimized owned Arrow/Parquet imports (PR #59, PR #61, PR #69).
- **Mathematical & Algorithmic Audit Remediations (Audits 1–15)**: Exact Efficiency Axiom TreeSHAP ($\sum \phi_i = f(x) - \mathbb{E}[f(X)]$) and constrained WLS KernelSHAP (`SwiftExplain`), Nelder-Mead simplex ETS parameter optimization and KalmanFilter covariance symmetry (`SwiftForecast`), DecisionTree entropy corrections and LinearRegression SVD pseudo-inverse (`SwiftML`), TargetEncoder empirical Bayes smoothing (`SwiftPreprocessing`), thread-safe vectorizers with `NSLock`, WordNet 3-color DFS taxonomy cycle validation, and calibrated VADER sentiment analysis (`SwiftNLP`) (PR #73–#78).
- **Comprehensive Test Suite**: 1,336 tests across 14 modules passing with 0 warnings under Swift 6 strict concurrency.

> For earlier release notes and historical changes (`v3.10.x` and earlier), see [CHANGELOG.md](CHANGELOG.md).

---

## 📚 Documentation & Resources

- **API Reference**: [Interactive DocC Documentation](https://nodibell.github.io/SwiftSci/) (100% verified coverage across all 15 modules)
- **Engineering Handbook & Practical Cookbook**: [SwiftSci 3.13.0 Complete Engineering Handbook & Practical Cookbook](https://github.com/Nodibell/SwiftSci/releases/tag/v3.13.0) ([PDF](https://github.com/Nodibell/SwiftSci/releases/download/v3.13.0/SwiftSci_Handbook_3.13.0.pdf) · [EPUB](https://github.com/Nodibell/SwiftSci/releases/download/v3.13.0/SwiftSci_Handbook_3.13.0.epub)) (118 pages, 67 verified recipes)
- **Benchmarking Suite**: [PERFORMANCE.md](PERFORMANCE.md) & [ACCURACY.md](ACCURACY.md)
- **Release History**: [CHANGELOG.md](CHANGELOG.md)
- **Roadmap**: [ROADMAP.md](ROADMAP.md)

---

## 📄 License

SwiftSci is available under the MIT License. See [LICENSE](LICENSE) for details.
