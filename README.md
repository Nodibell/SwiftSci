# SwiftSci 3.6.0

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
    .package(url: "https://github.com/Nodibell/SwiftSci.git", from: "3.6.0")
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
let score = vader.polarityScores(text: "SwiftSci 3.6.0 is incredibly fast, memory efficient, and robust!")
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
| **`SwiftAgent`** | **`MultiAgentOrchestrator`** with `AgentMessageBus` (`AsyncStream`), **`ReActAgent`** loop with structured timeouts, step-by-step lineage audit trails (`LineageRecord`), and sandboxed execution DSL. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftagent/) |
| **`SwiftVisualization`** | Native SwiftUI `Canvas` interactive charts (`SwiftSciChartView`), XSS-sanitized Plotly HTML export, and terminal ASCII/Braille graphs. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftvisualization/) |

---

## 📊 Performance & Memory Comparison (SwiftSci 3.6.0 vs Python)

All benchmarks are evaluated on **Apple Silicon (M-series, macOS 15 arm64)** with release builds (`-c release`) comparing SwiftSci directly against Python standard baselines (**NumPy, Pandas, Scikit-Learn, Statsmodels, SHAP**) using strictly equivalent data shapes, random seeds, and hyperparameters.

> 📖 **Complete Documentation:** See [PERFORMANCE.md](PERFORMANCE.md) for all 30+ benchmark scenarios and [ACCURACY.md](ACCURACY.md) for numerical accuracy verification.

| Domain / Scenario | SwiftSci 3.6.0 (Swift) | Python Baseline | Speedup | Winner | RAM Footprint (Swift vs Py) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ARIMA(1,1,1) Fit** (50k pts) | **`2.463 ms`** | `212.621 ms` (*Statsmodels*) | ⚡ **86.3×** | 🟢 **Swift** | **20 MB** vs 240 MB | Exact MLE recursion |
| **Holt-Winters Fit** (50k pts, s=12) | **`6.451 ms`** | `144.752 ms` (*Statsmodels*) | ⚡ **22.4×** | 🟢 **Swift** | **22 MB** vs 220 MB | Nelder-Mead optimization |
| **RandomForest Fit** (1k×4, 50 trees) | **`3.744 ms`** | `25.300 ms` (*Scikit-Learn*) | ⚡ **6.76×** | 🟢 **Swift** | **32 MB** vs 180 MB | Flat DOD tree buffers |
| **OneHotEncoder** (50k rows) | **`5.104 ms`** | `25.677 ms` (*Scikit-Learn*) | ⚡ **5.03×** | 🟢 **Swift** | **36 MB** vs 465 MB | SIMD categorical transform |
| **LIME Explain** (5 feats, 300 samples) | **`0.062 ms`** | `0.258 ms` (*Scikit-Learn*) | ⚡ **4.16×** | 🟢 **Swift** | **10 MB** vs 691 MB | ⚡ **200×** vs official `lime` pkg |
| **GBDT Regressor Fit** (1k×4, 50 est.) | **`8.023 ms`** | `32.366 ms` (*Scikit-Learn*) | ⚡ **4.03×** | 🟢 **Swift** | **32 MB** vs 190 MB | Flat DOD gradient boosting |
| **Two-Sample T-Test** (100k samples) | **`0.285 ms`** | `1.120 ms` (*SciPy*) | ⚡ **3.93×** | 🟢 **Swift** | **18 MB** vs 110 MB | Accelerate vDSP Welch's t-test |
| **IsolationForest Fit** (1k×10, 100 trees) | **`13.543 ms`** | `38.093 ms` (*Scikit-Learn*) | ⚡ **2.81×** | 🟢 **Swift** | **37 MB** vs 668 MB | Parallelized DOD outlier trees |
| **CNN Feature Extraction & Vision** | **`0.003 ms`** | `0.008 ms` (*NumPy*) | ⚡ **2.67×** | 🟢 **Swift** | **9 MB** vs 691 MB | Global pooling & Dice metric |
| **KernelSHAP Explain** (100 coalitions) | **`0.187 ms`** | `0.449 ms` (*SHAP*) | ⚡ **2.40×** | 🟢 **Swift** | **10 MB** vs 469 MB | Black-box Shapley attribution |
| **Classification ROC-AUC** (50k preds) | **`2.609 ms`** | `4.759 ms` (*Scikit-Learn*) | ⚡ **1.82×** | 🟢 **Swift** | **27 MB** vs 463 MB | Vectorized ranking algorithm |
| **VectorStore Cosine Search** (5k × 128d) | **`0.167 ms`** | `0.210 ms` (*NumPy*) | ⚡ **1.26×** | 🟢 **Swift** | **44 MB** vs 95 MB | In-memory Top-K retrieval |
| **CSV Read** (100k rows) | **`15.465 ms`** | `19.413 ms` (*Pandas*) | ⚡ **1.26×** | 🟢 **Swift** | **52 MB** vs 130 MB | POSIX memory-mapped parsing |
| **OneVsRestClassifier** (5 classes) | **`3.354 ms`** | `3.413 ms` (*Scikit-Learn*) | ⚡ **1.02×** | 🟢 **Swift** | **22 MB** vs 691 MB | 5-class Logistic OvR (31× less RAM) |
| **LinearSVC Fit** (1k×4, Metal GPU) | **`0.429 ms`** | `0.399 ms` (*Scikit-Learn*) | 0.93× | 🔴 **Python** | **37 MB** vs 668 MB | LibLinear C vs Metal GPU (18× less RAM) |
| **TF-IDF Vectorizer** (50 docs) | **`0.667 ms`** | `0.359 ms` (*Scikit-Learn*) | 0.54× | 🔴 **Python** | **22 MB** vs 691 MB | Sparse text vectorization |
| **TreeSHAP Explanation** (100 samples) | **`0.312 ms`** | `0.071 ms` (*SHAP*) | 0.23× | 🔴 **Python** | **11 MB** vs 691 MB | Lundberg C++ extension (62× less RAM) |
| **SQLite DataFrame Ingestion** | **`0.667 ms`** | `0.105 ms` (*Pandas*) | 0.16× | 🔴 **Python** | **11 MB** vs 691 MB | In-memory C-API (63× less RAM) |

### 🎯 Model Accuracy & Forecast Quality Scorecard

SwiftSci incorporates an automated validation scorecard confirming numerical parity against ground truth test datasets:

```text
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │                    MODEL ACCURACY & FORECAST QUALITY SCORECARD                     │
  ├────────────────────────────────────────────────────────────────────────────────────┤
  │ [Forecast] Holt-Winters (h=24) : RMSE=9.764, MAE=8.631, MAPE=6.11%, R²=-1.333      │
  │ [Forecast] ARIMA(1,1,1) (h=24) : RMSE=10.218, MAE=8.557, MAPE=5.87%, R²=-1.555     │
  │ [ML Reg]   GBDT (30 trees, d=4) : RMSE=0.421, MAE=0.344, R²=0.9879                 │
  │ [ML Cls]   RandomForest (30 tr.): Accuracy=99.00%, F1=0.991                        │
  │ [NLP Cls]  NaiveBayes (3-class) : Accuracy=35.00%, Macro-F1=0.342                  │
  └────────────────────────────────────────────────────────────────────────────────────┘
```

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
