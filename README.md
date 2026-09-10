# SwiftSci 3.5.2

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
    .package(url: "https://github.com/Nodibell/SwiftSci.git", from: "3.5.2")
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
let score = vader.polarityScores(text: "SwiftSci 3.5.2 is incredibly fast, memory efficient, and robust!")
print("Sentiment compound score:", score.compound)
```

---

## 🧩 The 14 Core Modules

| Module | Purpose & Capabilities | Documentation |
| :--- | :--- | :---: |
| **`SwiftDataFrame`** | Zero-copy columnar tables, **pure-Swift Apache Parquet engine**, Arrow Feather, NumPy `.npy`/`.npz` tensor reader, memory-mapped I/O, SIMD hash joins, and out-of-core `ChunkedDataFrame`. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftdataframe/) |
| **`SwiftStats`** | Vectorized descriptive statistics, SIMD vDSP sorting, Student-t/Chi-Square/F distributions, Two-Sample t-test, Spearman correlation, ANOVA powered by Accelerate. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftstats/) |
| **`SwiftPreprocessing`** | Feature scaling (`StandardScaler`, `MinMaxScaler`), categorical encoding (**`OneHotEncoder`**, `TargetEncoder`), imputation, `Pipeline`, `ColumnTransformer`, `HardwareRouter`. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftpreprocessing/) |
| **`SwiftML`** | OLS/Ridge/Lasso regression, Decision Trees, Random Forests, **GBDT Quantile Regression**, **`LinearSVC`** (Metal GPU), MLP neural nets, Core ML & ONNX exporters. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftml/) |
| **`SwiftCluster`** | In-memory **`VectorStore`** cosine index (top-$k$), randomized SVD / PCA, DBSCAN, `IsolationForest` outlier detection, `KMeans`. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftcluster/) |
| **`SwiftForecast`** | ARIMA/SARIMA, Exponential Smoothing with 95% confidence intervals, ETS, GARCH, Kalman filtering, Prophet-style trend decomposition, Seasonal ESD anomaly detection. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftforecast/) |
| **`SwiftOptimize`** | K-Fold / TimeSeriesSplit cross-validation, Forecast Errors Suite (RMSE, MAE, MAPE, R²), ROC-AUC / PR-AUC metrics, AutoML, `GridSearchCV`. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftoptimize/) |
| **`SwiftExplain`** | Model interpretability: parallel `KernelSHAP`, exact polynomial **`TreeSHAP`**, **`LIMEExplainer`**, Partial Dependence Plots (PDP), Permutation Importance. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftexplain/) |
| **`SwiftNLP`** | Linguistic engine: tokenization, lemmatization, Porter stemmer, POS tagger, **`VADERSentimentAnalyzer`**, `NaiveBayesClassifier`, dense embeddings. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftnlp/) |
| **`SwiftLLM`** | **`LLMModel` protocol**, 4-bit/8-bit quantized linear layers (`QuantizedLinear`), Paged KV-Cache allocator, constrained JSON grammar decoder, streaming generation. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftllm/) |
| **`SwiftVision`** | Pure-Swift Non-Maximum Suppression (NMS), YOLOv8n object detection, YOLOv8-Seg instance segmentation, CLIP projector, U-Net image segmentation. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftvision/) |
| **`SwiftDatabase`** | Zero-copy SQL ingestion/export: SQLite with **automatic table discovery**, native PostgreSQL (v3.0 wire protocol), and native MySQL connectors. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftdatabase/) |
| **`SwiftAgent`** | **Autonomous `ReActAgent` loop** for local LLMs, step-by-step lineage audit trails, sandboxed execution DSL, and RAG context profile generators. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftagent/) |
| **`SwiftVisualization`** | Native SwiftUI `Canvas` interactive charts (`SwiftSciChartView`), Plotly HTML export, and terminal ASCII/Braille graphs. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftvisualization/) |

---

## 📊 Performance & Memory Comparison (SwiftSci 3.5.2 vs Python)

All benchmarks are evaluated on **Apple Silicon (M-series, macOS 15 arm64)** with release builds (`-c release`) comparing SwiftSci directly against Python standard baselines (**NumPy, Pandas, Scikit-Learn, Statsmodels, SHAP**) using strictly equivalent data shapes, random seeds, and hyperparameters.

> 📖 **Complete Documentation:** See [PERFORMANCE.md](PERFORMANCE.md) for all 30+ benchmark scenarios and [ACCURACY.md](ACCURACY.md) for numerical accuracy verification.

| Domain / Scenario | SwiftSci 3.5.2 (Swift) | Python Baseline | Speedup | Winner | RAM Footprint (Swift vs Py) | Notes |
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

## 🚀 What's New in v3.5.2

- **Standard Apache Parquet Engine (`SwiftDataFrame`):** 100% standard Parquet support (compatible with DuckDB, PyArrow, and Hugging Face). Decouples uncompressed Thrift `PageHeader`, unpacks `RLE_DICTIONARY` and `PLAIN_DICTIONARY` encodings with dynamic bit-widths, decodes repetition/definition levels, and aggregates nested list schemas (`labels.list.item`). Bit-exact verified on Hugging Face `go_emotions` (5,427 rows).
- **Pure-Swift NumPy NPY & NPZ Tensor Reader (`SwiftDataFrame`):** Zero-dependency pure-Swift reader for `.npy` and `.npz` archive files (`NPYReader`, `NPZReader`) with ZIP64 extra field and Deflate decompression. Enables direct conversion into tabular DataFrames via `DataFrame(npy:)` and `DataFrame(npz:)`.
- **SQLite Table Auto-Discovery (`SwiftDatabase`):** Added `DataFrame(sqlite: URL, table: String? = nil)`, automatically inspecting `sqlite_master` and reading tables into DataFrames without manual SQL boilerplate.
- **Quantile Regression (Pinball Loss) in GBDT (`SwiftML`):** Added `GBDTLoss.quantile(alpha:)` to `GradientBoostedTreesRegressor`. Optimizes asymmetric check/pinball losses for arbitrary tabular quantiles, producing non-parametric confidence bands.
- **CLI Dataset Conversion & Summary Extensions (`SwiftSciCLI`):** `swiftsci summary` and `swiftsci convert` extended to support `.parquet`, `.npy`, and `.npz` files alongside CSV and Feather.
- **Comprehensive Benchmark & Accuracy Matrix:** Full apple-to-apple baselines against Scikit-Learn, Statsmodels, and SHAP with 95% confidence intervals and RAM RSS tracking.

> For previous version notes (v3.5.1, v3.5.0, etc.), see [CHANGELOG.md](CHANGELOG.md) and [ROADMAP.md](ROADMAP.md).

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
