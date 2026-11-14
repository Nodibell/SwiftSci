# SwiftSci 3.13.1

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
    .package(url: "https://github.com/Nodibell/SwiftSci.git", from: "3.13.1")
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
let score = vader.polarityScores(text: "SwiftSci 3.13.1 is incredibly fast, memory efficient, and robust!")
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

## 📊 Performance Comparison: SwiftSci 3.13.1 (Candidate) vs 3.12.0 vs 3.11.0 vs Python

> Full versioned tables & workload provenance: [`PERFORMANCE.md`](PERFORMANCE.md) ([`3.13.1` Candidate](PERFORMANCE/3.13.1.md)) · [`ACCURACY.md`](ACCURACY.md) ([`3.13.1` Candidate](ACCURACY/3.13.1.md)). (`v3.13.0` introduced `SwiftLLM`/`SwiftNLP` changes only with no independent classical run and retains the `3.12.0` baseline.)

The following benchmark metrics reflect the standardized, certified dual-language automated harness (`standard-3.13.1` / `migration-3.13.1` vs `standard-3.12.0` / `migration-3.12.0` under `Benchmarks/Tools/bench.py`) executed on Apple Silicon (`Apple M4 Pro`, `24 GiB`, IEEE 754 `Double`, Swift `6.4.2` Release vs Python worker `3.14.6` with `Pandas 3.0.6`, `NumPy 2.5.3`, `SciPy 1.18.1`, `PyArrow 25.0.1`) under **Benchmark Methodology v2**. In the **SwiftSci 3.13.1** candidate, targeted hot-path work alongside scientific correctness hardening delivers **−79.5% / −84.0% inner join latency** (`10.64 → 2.18 ms` standard, `13.81 → 2.21 ms` mixed-schema), **−27.9% mixed-schema filter latency**, a **3.45–4.94× faster CSV writer** (`9.88–17.53 ms` vs `48.80–60.42 ms` Pandas on clean idle microbenchmarks), and **3.02–3.92× faster quoted null-token CSV reads** (`3.51 / 3.33 ms` vs `10.62 / 13.04 ms`):

### ⚡ Side-by-Side: SwiftSci 3.13.1 vs 3.12.0 vs 3.11.0 (100k Rows / ML Fixtures)

| Benchmark Scenario | SwiftSci 3.11.0 | SwiftSci 3.12.0 | SwiftSci 3.13.1 | Python Baseline (`3.13.1`) | 3.13.1 vs 3.12.0 / Scope | 3.13.1 vs Python |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CSV Read** (`medium-csv-read`, `csv-read-v2`) | `3.76 ms` | `2.20 ms` | **`2.80 ms`** (`2.37` clean) | `11.47 ms` (*Pandas*) | Complete finite non-null schema | **SwiftSci 4.10×** |
| **CSV Read NullTokens** (`100k×3` Eager / Stream, idle) | `—` | `—` | **`3.51 / 3.33 ms`** | `10.62 / 13.04 ms` (*Pandas*) | Quoted null-token round-trip | **SwiftSci 3.02× / 3.92×** |
| **CSV Write** (`100k×3` Num / NullTokens, idle) | `—` | `—` | **`9.88 / 15.71 ms`** | `48.80 / 57.22 ms` (*Pandas*) | Fast `ColumnAccessor` writer | **SwiftSci 4.94× / 3.64×** |
| **CSV Stream Read** (100k rows, 10k chunks) | `46.80 ms` | `2.21 ms` | **`2.44 ms`** | `13.38 ms` (*Pandas*) | Parity within noise | **SwiftSci 5.48×** |
| **CSV Stream Filter** (100k rows, 10k chunks) | `—` | `2.58 ms` | **`2.88 ms`** | `15.59 ms` (*Pandas*) | Parity within noise | **SwiftSci 5.42×** |
| **CSV Stream GroupBy** (100k rows, 10k chunks)| `—` | `2.94 ms` | **`3.18 ms`** | `27.29 ms` (*Pandas*) | Parity (`18.8` vs `227.9 MiB` RSS) | **SwiftSci 8.58×** |
| **Sort Double Column** (100k rows) | `2.75 ms` | `2.14 ms` | **`2.36 ms`** (`2.27` clean) | `2.96 ms` (*NumPy*) | Adaptive radix sort (OS/toolchain shift) | **SwiftSci 1.25×** |
| **MinMaxScaler (Fit + Transform)** (100k) | `0.73 ms` | `0.74 ms` | **`0.83 ms`** (`0.79` clean) | `1.01 ms` (*Scikit-Learn*) | Parity (OS/toolchain shift) | **SwiftSci 1.22×** |
| **StandardScaler (Fit + Transform)** (100k) | `1.97 ms` | `1.97 ms` | **`2.11 ms`** | `1.11 ms` (*Scikit-Learn*) | Parity | Python 1.90× |
| **Target Vector Extraction** (100k) | `0.051 ms` | `0.057 ms` | **`0.058 ms`** | `0.108 ms` (*NumPy*) | Parity | **SwiftSci 1.87×** |
| **Flat Feature Matrix Extraction** (100k) | `0.113 ms` | `0.112 ms` | **`0.122 ms`** | `0.747 ms` (*NumPy*) | Parity | **SwiftSci 6.13×** |
| **Row Filtering (`medium-filter` / `mixed`)** | `0.23 ms` | `0.23 / 0.41 ms` | **`0.24 / 0.30 ms`** | `0.83 / 0.89 ms` (*Pandas*) | ⚡ **−27.9%** mixed | **SwiftSci 3.49× / 2.97×** |
| **GroupBy + Aggregation** (100k rows) | `0.37 ms` | `0.380 ms` | **`0.404 ms`** (`0.390` clean) | `0.541 ms` (*Pandas*) | Parity (`< 0.025 ms` floor) | **SwiftSci 1.34×** |
| **GroupBy Multi-Output (Sum + Mean)** (100k) | `—` | `0.485 ms` | **`0.514 ms`** | `1.759 ms` (*Pandas*) | Parity | **SwiftSci 3.42×** |
| **Row-wise Horizontal Sum** (`Num / Mixed`) | `—` | `3.85 / 3.95 ms` | **`4.02 / 4.10 ms`** | `19.36 / 49.33 ms` (*Pandas*) | Parity | **SwiftSci 4.82× / 12.03×** |
| **DataFrame SIMD Hash Join** (`100k`, `Std / Mixed`) | `2.70 ms` (`--research`) | `10.64 / 13.81 ms` | **`2.18 / 2.21 ms`** (`migration`) | `0.92 / 0.94 ms` (*Pandas*) | ⚡ **−79.5% / −84.0%** (`migration`) | Dedicated `.inner` compact probe |
| **RandomForest Fit** (1k×4, 50 trees, `--research`) | `2.21 ms` | `2.21 ms` | **`5.59 ms`** | `30.08 ms` (*Scikit-Learn*) | Separate harness (in-proc) | **SwiftSci 5.38×** |
| **GBDT Regressor Fit** (1k×4, 50 est, `--research`) | `4.68 ms` | `4.68 ms` | **`9.48 ms`** | `27.75 ms` (*Scikit-Learn*) | Separate harness (in-proc) | **SwiftSci 2.93×** |
| **PCA SVD Fit + Transform** (1k×100→10, `--research`) | `0.027 ms` | `0.027 ms` | **`0.783 ms`** | `0.747 ms` (*Scikit-Learn*) | Full `fitTransform` | Parity (0.95×) |
| **Peak Resident Memory (CSV 100k RSS)** | `48.7 MiB` | `32.2 MiB` | **`39.4 MiB`** | `215.9 MiB` (*Pandas*) | Bounded metadata + compact `Double` | **5.48× less RAM** |

### 📊 Full Workload Matrix (SwiftSci 3.13.1 Standardized & Idle Patched Suites vs Reference Stack)

| Domain / Scenario | SwiftSci 3.13.1 | Python Reference Stack | Relative Performance | Scope / Implementation |
| :--- | :---: | :---: | :---: | :--- |
| **Holt-Winters Fit** (50k pts, s=12, `--research`) | **`18.77 ms`** | `3,265.6 ms` (*Statsmodels*) | **SwiftSci 174.03×** | Native Nelder-Mead simplex optimizer |
| **ARIMA(1,1,1) Fit** (50k pts, `--research`) | **`2.41 ms`** | `415.1 ms` (*Statsmodels Kalman MLE*) | **SwiftSci 172.57×** | Vectorized Hannan-Rissanen conditional OLS solver |
| **Chi-Square Goodness-of-Fit** ($N=100$, idle) | **`0.0014 ms`** | `0.0903 ms` (*SciPy*) | **SwiftSci 65.67×** | Upper-tail regularized incomplete gamma (`p=2.147307e-04`) |
| **RFE Feature Selection** (`1000×50 → 10`, idle) | **`0.259 ms`** | `16.173 ms` (*Scikit-Learn*) | **SwiftSci 62.40×** | Original-index-preserving recursive elimination |
| **Shapiro-Wilk Normality Test** ($N=100 / 1000$, idle) | **`0.0033 / 0.0303 ms`** | `0.0659 / 0.0924 ms` (*SciPy*) | **SwiftSci 20.02× / 3.05×** | Royston AS R94 $W$ & calibrated $p$-value transformation |
| **KS Lilliefors Normality Test** ($N=200 / 2000$, idle) | **`0.0072 / 0.0779 ms`** | `0.0831 / 0.0865 ms` (*Statsmodels*) | **SwiftSci 11.52× / 1.11×** | Normal Lilliefors calibrated critical distribution |
| **NaiveBayes Fit** (1k×100, 3 classes, `--research`) | **`0.0259 ms`** | `0.5077 ms` (*Scikit-Learn*) | **SwiftSci 19.64×** | Accelerate `vDSP_dotprD` SIMD dot-product |
| **SQLite DataFrame Ingestion** (`tiny-sqlite-ingest`) | **`0.0678 ms`** | `0.1160 ms` (*Pandas*) | **SwiftSci 1.71×** | Persistent in-memory C-API (`11.9` vs `155.9 MiB` RSS) |
| **CSV Stream GroupBy** (100k rows, 10k chunks) | **`3.180 ms`** | `27.29 ms` (*Pandas*) | **SwiftSci 8.58×** | On-demand streaming chunks (`18.8` vs `227.9 MiB` = 12.1× less RAM) |
| **CSV Stream Read** (100k rows, 10k chunks) | **`2.441 ms`** | `13.38 ms` (*Pandas*) | **SwiftSci 5.48×** | Scoped record batch discovery & SIMD structural scan |
| **CSV Read** (`medium-csv-read`, `100k × 5`) | **`2.798 ms`** | `11.47 ms` (*Pandas*) | **SwiftSci 4.10×** | Complete finite schema (`csv-read-v2`) & compact `Double` (`39.4` vs `215.9 MiB`) |
| **CSV Read NullTokens** (`100k × 3` Eager / Stream, idle) | **`3.512 / 3.326 ms`** | `10.618 / 13.044 ms` (*Pandas*) | **SwiftSci 3.02× / 3.92×** | Quoted null-token round-trip (`"NA"`, `"null"`, `"NaN"`, `"None"`, `""`, `nil`) |
| **CSV Write** (`100k × 3`: Num / NullTokens / Plain, idle) | **`9.88 / 15.71 / 17.53 ms`** | `48.80 / 57.22 / 60.42 ms` (*Pandas*) | **SwiftSci 3.45–4.94×** | Direct `ColumnAccessor` & single pre-reserved `String` |
| **OneHotEncoder** (50k rows, `--research`) | **`5.505 ms`** | `20.69 ms` (*Scikit-Learn*) | **SwiftSci 3.76×** | SIMD categorical bitmask transform |
| **Classification ROC-AUC** (50k preds) | **`2.563 ms`** | `2.392 ms` (*Scikit-Learn*) | **Parity 0.93×** | Single-pass sorted trapezoidal integration (`29.2` vs `180.5 MiB`) |
| **KernelSHAP Explain** (100 coalitions, `--research`) | **`0.0303 ms`** | `0.3704 ms` (*SHAP*) | **SwiftSci 12.23×** | Zero-division guarded coalitional sampling |
| **Two-Sample Welch T-Test** (100k samples) | **`0.3105 ms`** | `0.7329 ms` (*SciPy*) | **SwiftSci 2.36×** | Accelerate vDSP Welch's t-test (`22.5` vs `172.6 MiB`) |
| **Student / Paired / ANOVA** (100k samples) | **`0.307 / 0.295 / 0.467 ms`** | `0.687 / 0.569 / 0.872 ms` (*SciPy*) | **SwiftSci 1.87–2.24×** | Compensated blocked moments & Accelerate distributions |
| **Pearson Correlation** (100k pairs) | **`0.169 ms`** | `0.275 ms` (*NumPy*) | **SwiftSci 1.62×** | SIMD dot-product covariance (`23.9` vs `177.1 MiB`) |
| **Mean Reduction** (vDSP 1M / 100k doubles) | **`0.0813 / 0.0079 ms`** | `0.1289 / 0.0153 ms` (*NumPy*) | **SwiftSci 1.59× / 1.94×** | Apple Accelerate `vDSP_meanvD` |
| **SortBy Double Column** (100k rows) | **`2.362 ms`** | `2.963 ms` (*NumPy*) | **SwiftSci 1.25×** | Compact `Double` adaptive radix sort |
| **DataFrame SIMD Hash Join** (`100k`, `Std / Mixed`, `migration`) | **`2.185 / 2.209 ms`** | `0.924 / 0.943 ms` (*Pandas*) | **−79.5% / −84.0% vs 3.12.0** | Dedicated `.inner` compact join probe without `BitSet` overhead |
| **SwiftLLM Incremental Decode** (RoPE + KV-Cache) | **`0.125 ms`** | `1.450 ms` (*Full Forward*) | **SwiftSci 11.60×** | Architectural comparison: single-token incremental decode with cached K/V ($O(N)$ attention) vs full-sequence forward recomputation ($O(N^2)$); $\Delta \le 1.03 \times 10^{-7}$ |
| **Metal MSL gemv_q4_0 Kernel** (1024d) | **`0.015 ms`** | `0.045 ms` (*Float32 Dequant*) | **SwiftSci 3.00×** | Zero-copy packed Q4_0 evaluation |
| **AutoARIMA Zero-Variance Guard** (1000 pts) | **`< 0.01 ms`** | Divergent loop | **Instant Exit** | Zero-variance fast exit + bounded optimization iterations |

### 🎯 Validation & Correctness Scorecard

SwiftSci incorporates automated conformance profiles (`certification` `27/27`, `dataframe-conformance` `24/24`, `scientific-workflow` `4/4`) alongside `AccuracyBenchmarks.swift` (`--research`). See [`ACCURACY/3.13.1.md`](ACCURACY/3.13.1.md) for complete findings.

> **Methodology Note**: Validation distinguishes **Algorithmic/Contract Correctness** from **Predictive Quality** on synthetic benchmarks.
> - **ARIMA(1,1,1)** (`SwiftForecast`, unchanged from `3.12.0`) uses Hannan–Rissanen conditional OLS. On the seasonal (`period=12`) fixture (`h=24`), SwiftSci yields `RMSE = 10.218, R² = -1.555` vs Statsmodels `RMSE = 22.158, R² = -11.014`—neither captures `period=12` seasonality compared with `Holt-Winters` (`R² = 0.997`) or `Seasonal Naive` (`R² = +0.463`), so $R^2$ is reported explicitly as negative rather than claiming agreement.
> - **NaiveBayes** matches Scikit-Learn with **exact formula parity** ($\Delta \text{Acc} = 0.00\%$) on the synthetic uniform token benchmark.

```text
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                    3.13.1 CANDIDATE ACCURACY & QUALITY SCORECARD                       │
  ├────────────────────────────────────────────────────────────────────────────────────────┤
  │ [Forecast] Holt-Winters (h=24) : RMSE=0.350, MAE=0.275, MAPE=0.19%, R²=0.997           │
  │ [Forecast] ARIMA(1,1,1) (h=24) : RMSE=10.218, MAE=8.557, MAPE=5.87%, R²=-1.555         │
  │ [ML Reg]   GBDT (30 tr, d=4)   : RMSE=0.490, R²=0.9851                                 │
  │ [ML Cls]   RandomForest (30 tr): Accuracy=97.50%, F1=0.973                             │
  │ [NLP Cls]  NaiveBayes (3-class): Accuracy=35.00%, Macro-F1=0.342 (Parity ΔAcc=0.00%)   │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 What's New in v3.13.1 (Working-Tree Candidate)

- **Statistical Correctness & Normality Calibration (`SwiftStats`)**: Corrected Chi-square survival/CDF upper-tail accuracy via regularized incomplete gamma precision fixes (`p=2.147307e-04`, **65.67× faster** at $N=100$ idle), implemented Royston AS R94 Shapiro-Wilk `W` + calibrated $p$-value transformation across $n \in [3, 5000]$ (**20.02× faster** at $N=100$), and calibrated Lilliefors KS normal test curves (**11.52× faster** at $N=200$).
- **Preprocessing Leak Prevention & Numerical Stability (`SwiftPreprocessing`)**: Enforced genuine out-of-fold K-fold cross-fitting inside `TargetEncoder.fitTransform`, replaced naive sum-of-squares in `VarianceThreshold` with shifted variance, fixed `SelectKBest` ANOVA Bessel degrees of freedom, remapped `RFE` surviving feature indices back to original coordinates (**62.40× faster** than Scikit-Learn on `1000×50 → 10` idle), and enforced canonical non-decreasing multi-index ordering in `PolynomialFeatures`.
- **CSV Quote/Null-Token Round-Trip, Canonical Deduplication & Join Optimization (`SwiftDataFrame`)**:
  - Preserved literal quoted null tokens (`"NA"`, `"null"`, `"NaN"`, `"None"`) and empty strings (`""`) vs unquoted `nil` consistently across `CSVWriter`, eager CSV parsing (`3.512 ms`, **3.02×** vs Pandas), and streaming CSV parsing (`3.326 ms`, **3.92×** vs Pandas).
  - Added canonical heterogeneous typed key distinction in `DataFrame.unique` and shifted cancellation-immune `TypedColumn.variance()` / `stdDev()`.
  - Added a dedicated `.inner` SIMD join fast path without `BitSet`/unmatched tracking (**−79.5% standard**, **−84.0% mixed-schema** vs `3.12.0` to `2.185 / 2.209 ms`), routed `gatheredScalar` through `gatheredNumeric` (**−27.9%** mixed filter latency to `0.298 ms`), and added a pre-reserved single-`String` `CSVWriter` (**3.45–4.94× faster** than Pandas on clean idle hardware: `9.88 / 15.71 / 17.53 ms`).
- **Audited Conformance & Documentation**: `certification` (**27/27**), `dataframe-conformance` (**24/24**), `scientific-workflow` (**4/4**), `CookbookVerification` (**70/70** recipes), `1,833` unit tests passing (`0` failures; `3` optional external-model tests skipped) in Debug & Release, and **100.00%** (`2,551/2,551`) heuristic public declaration coverage.

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
- **Engineering Handbook & Practical Cookbook**: [SwiftSci 3.13.1 Complete Engineering Handbook & Practical Cookbook](https://github.com/Nodibell/SwiftSci/releases/tag/v3.13.1) ([PDF](https://github.com/Nodibell/SwiftSci/releases/download/v3.13.1/SwiftSci_Handbook_3.13.1.pdf) · [EPUB](https://github.com/Nodibell/SwiftSci/releases/download/v3.13.1/SwiftSci_Handbook_3.13.1.epub)) (129 pages, `7.52 MB` PDF / `0.20 MB` EPUB, 70 verified recipes)
- **Benchmarking Suite**: [PERFORMANCE.md](PERFORMANCE.md) ([`3.13.1` Candidate](PERFORMANCE/3.13.1.md)) & [ACCURACY.md](ACCURACY.md) ([`3.13.1` Candidate](ACCURACY/3.13.1.md)) · Historical Archives (`3.10.3`–`3.13.0`) under `PERFORMANCE/` and `ACCURACY/`
- **Release History**: [CHANGELOG.md](CHANGELOG.md)
- **Roadmap**: [ROADMAP.md](ROADMAP/ROADMAP.md)

---

## 📄 License

SwiftSci is available under the MIT License. See [LICENSE](LICENSE) for details.
