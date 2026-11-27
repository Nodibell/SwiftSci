# SwiftSci 3.14.0

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
- **Compact numeric storage:** Typed columns and reusable batches reduce intermediate conversions and allocations in supported workflows.
- 🧩 **Zero-Boilerplate Ingestion**: Pure-Swift readers for Apache Parquet, Apache Arrow Feather, NumPy `.npy`/`.npz`, and automatic SQLite schema discovery.

---

## 📦 Installation

Add SwiftSci to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/Nodibell/SwiftSci.git", from: "3.14.0")
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

> **Platform & Toolchain Requirements:** macOS 14+ (Apple Silicon M-Series), Swift 6.3+ (use an Xcode toolchain that provides Swift 6.3 or later).

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

// 3. Categorical encoding
let ohe = OneHotEncoder()
try ohe.fit([["engineering"], ["finance"], ["engineering"]])
let encoded = try ohe.transform([["engineering"], ["finance"]])

// 4. Black-Box Model Explainability (TreeSHAP & LIME)
let lime = LIMEExplainer(kernelWidth: 0.75, regularization: 0.01)
let explanation = await lime.explain(model: { $0.reduce(0.0, +) }, instance: [1.0, 2.0, 3.0], numSamples: 300)

// 5. Native Sentiment Analysis
let vader = VADERSentimentAnalyzer()
let score = vader.polarityScores(text: "SwiftSci makes scientific computing accessible in Swift.")
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
| **`SwiftExplain`** | Model interpretability: parallel `KernelSHAP` with zero-division guards and exact paths for $M \le 2$, **`TreeSHAP`** with exact enumeration for up to 12 split features and a seeded approximation for larger inputs, **`LIMEExplainer`**, Partial Dependence Plots (PDP), Permutation Importance. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftexplain/) |
| **`SwiftNLP`** | Linguistic engine: byte-level UTF-8 invertible **`BPETokenizer`**, lemmatization, Porter stemmer, POS tagger, **`VADERSentimentAnalyzer`**, `NaiveBayesClassifier`, dense embeddings. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftnlp/) |
| **`SwiftLLM`** | **`LLMModel` protocol**, custom Metal MSL SIMD-group quantization kernels (`QuantizedGEMM.metal`), 4-bit/8-bit quantized linear layers (`QuantizedLinear`), Paged KV-Cache allocator, constrained JSON grammar decoder, streaming generation. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftllm/) |
| **`SwiftVision`** | Zero-heap hardware-accelerated **`BoundingBoxSIMD`** NMS, YOLOv8n object detection, YOLOv8-Seg instance segmentation, CLIP projector, U-Net image segmentation. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftvision/) |
| **`SwiftDatabase`** | Zero-copy SQL ingestion/export: SQLite with **1024-row column buffer ingestion**, auto-discovery, and C-API read-only query mode; native PostgreSQL (`SCRAM-SHA-256` / `MD5`) and MySQL (`mysql_native_password` SHA-1; unsupported plugins fail closed) wire connectors with fail-closed `SSLMode.require` protection. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftdatabase/) |
| **`SwiftAgent`** | Omni-module autonomous agents: **`SwiftSciToolbox`** scientific tools, typed **`AgentToolV2`** with JSON schemas, **`ReActAgent`** with real-time async event streaming, **`SemanticVectorMemory`** (HNSW graph recall), and native SwiftUI **`AgentDialogueController`** (`@Observable`). | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftagent/) |
| **`SwiftVisualization`** | Native SwiftUI `Canvas` interactive charts (`SwiftSciChartView`), XSS-sanitized Plotly HTML export, and terminal ASCII/Braille graphs. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftvisualization/) |
| **`SwiftWorkflow`** | End-to-end persisted preprocessing, PCA, and classification pipelines (`PCAClassificationWorkflow`) with versioned JSON artifact serialization and strict feature schema validation. | [📖 DocC](https://nodibell.github.io/SwiftSci/documentation/swiftworkflow/) |

---

## What’s new in v3.14.0

- **`TreeSHAP` for GBDT & Per-Class Probabilities (`SwiftExplain`, `SwiftML`, `G-035`, `G-036`)**: Exact and sampled `TreeSHAP` for `GradientBoostedTreesRegressor`, per-class probability `TreeSHAP` (`explainClassDetailed`, `explainAllClassesDetailed`, `explainBatchClass`) for `DecisionTreeClassifier` and `RandomForestClassifier`, `TreeNodeWeighting` (`.uniformLeaf` / `.empiricalCover`), `ForestProbabilityMode` (`.softLeafDistribution` / `.hardVote`), and `FlatTreeNode.cover` / `classDistribution` persistence—verified to `<= 1.78e-15` against Python `shap.TreeExplainer` (`shap 0.54.1`).
- **Chat-Templated Agents, Stop Sequences & `GenerationConfiguration` (`SwiftAgent`, `SwiftLLM`, `SwiftNLP`, `G-038`)**: `ReActAgent.stream(query:chatTemplate:...)`, `ChatTemplate.defaultStopSequences` across all 6 families, `LLMOptions.stopSequences` / `stopTokenIDs`, and `GenerationConfiguration` (`generation_config.json`) support in `TransformerDecoder`.
- **`TrainedVisionClassifier`, `OneVsRestClassifier` & Core ML Vision Pipelines (`SwiftVision`, `SwiftML`, `G-039`, `G-040`)**: `YOLODatasetMetadata`, image/background crop helpers, `ImageDataset.loadFolder` / `loadSplitFolder`, `OneVsRestClassifier` `ClassifierEstimator` conformance (**2.94×** faster `predictProbability`: `0.069 ms` vs scikit-learn `0.203 ms`), `CoreMLExporter.exportVisionClassifierPipeline`, and `TrainedVisionClassifier` with compiled Core ML `VisionFeaturePrint.Scene` parity.
- **AC-powered 4-engine benchmarking**: `6/6` `production-default` comparison campaigns (`10,300` validated samples across `swiftsci`, `pandas`, `polars`, and `duckdb`), `39/39` scientific validation workloads, `1,912/1,919` Swift tests (`0` failures, `7` env-gated skips), `2,641/2,641` DocC declarations, and `71/71` Cookbook recipes.

See the [performance report](PERFORMANCE/3.14.0.md) and [accuracy report](ACCURACY/3.14.0.md). The [changelog](CHANGELOG.md) covers earlier releases.

## 📚 Documentation & Resources

- **API Reference**: [Interactive DocC Documentation](https://nodibell.github.io/SwiftSci/)
- **SwiftSci 3.14.0 Handbook & Cookbook:** [Handbook PDF](https://github.com/Nodibell/SwiftSci/releases/download/v3.14.0/SwiftSci_Handbook_3.14.0.pdf) · [Handbook EPUB](https://github.com/Nodibell/SwiftSci/releases/download/v3.14.0/SwiftSci_Handbook_3.14.0.epub) · [Cookbook PDF](https://github.com/Nodibell/SwiftSci/releases/download/v3.14.0/SwiftSci_Cookbook_3.14.0.pdf) · [Cookbook EPUB](https://github.com/Nodibell/SwiftSci/releases/download/v3.14.0/SwiftSci_Cookbook_3.14.0.epub) · [Companion Guide](Documentation/Handbook-3.14.0.md)
- **Benchmarks:** [Performance](PERFORMANCE/3.14.0.md) · [Accuracy](ACCURACY/3.14.0.md) · [Performance archive](PERFORMANCE.md) · [Accuracy archive](ACCURACY.md).
- **Release History**: [CHANGELOG.md](CHANGELOG.md)
- **Roadmap**: [ROADMAP.md](ROADMAP/ROADMAP.md)

---

## 📄 License

SwiftSci is available under the MIT License. See [LICENSE](LICENSE) for details.
