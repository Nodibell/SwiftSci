# SwiftSci 3.5.0

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

## What's New in 3.5.0

- **Multi-Round Statistical Benchmark System (`SwiftSciBenchmarks`):** Scientific benchmark harness with $R \times I$ multi-round sampling, **95% Confidence Intervals** ($\text{Margin of Error} = 1.96 \cdot \frac{s}{\sqrt{N}}$), **20% Trimmed Mean**, Median, and live **Resident Memory (RAM RSS MB)** profiling via Mach task basic info.
- **High-Speed OneHotEncoder (5.03× vs Scikit-Learn):** SIMD categorical encoder processing 50k rows in **5.10 ms** (vs 25.68 ms in Python), with **13× lower RAM footprint** (36 MB vs 465 MB).
- **Sub-Millisecond Forecast & Regression Error Metrics Suite:** Vectorized implementations of `RMSE`, `MAE`, `MAPE`, and $R^2$ executing 100k data points in **0.84 ms** via Accelerate `vDSP`.
- **Accelerated Classification ROC-AUC (1.82× vs Scikit-Learn):** Optimized rank-based Area Under the ROC Curve computing 50k predictions in **2.61 ms** (vs 4.76 ms in Python).
- **Pure-Swift Typed Parquet Snappy Engine & SIMD Hash Join (`SwiftDataFrame`):** Zero-alloc direct typed column fast-paths (`TypedColumn<Int64>`, `TypedColumn<Double>`, `TypedColumn<String>`) and LZ77 fast-skip compression step jumping.
- **Two-Sample T-Test & Spearman Rank Correlation (`SwiftStats`):** Welch's Two-Sample T-Test executing 100k samples in **0.285 ms** (3.93× faster than SciPy `ttest_ind`).
- **In-Memory VectorStore Cosine Index (`SwiftCluster`):** Zero-copy batch vector indexing and top-$k$ nearest neighbor search executing $5,000 \times 128\text{d}$ vectors in **0.167 ms**.
- **VADER Sentiment Analysis & NaiveBayes Classifier (`SwiftNLP`):** Full 7,500+ rule lexicon sentiment analysis (1k sentences in **2.76 ms**) and Laplace-smoothed `NaiveBayesClassifier` (1k docs in **3.79 ms**).
- **100% DocC API Coverage:** Maintained 100.00% public API documentation coverage across all 14 modules.

See [CHANGELOG.md](CHANGELOG.md).

---

## Core Modules

| Module | Description | Docs |
| :--- | :--- | :---: |
| **`SwiftDataFrame`** | SIMD vectorised `filterFast`, Accelerate Double sorting, Arrow zero-copy Feather (`FeatherReader`/`FeatherWriter`), **pure-Swift Parquet engine** (`ParquetReader`/`ParquetWriter`), out-of-core streaming **`ChunkedDataFrame`**, **`MemoryMappedReader`**, and SIMD hash joins. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftdataframe/) |
| **`SwiftStats`** | Vectorized descriptive statistics, SIMD vDSP sorting, Student-t/Chi-Square/F distributions, Two-Sample t-test, Spearman correlation, ANOVA powered by `Accelerate vDSP`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftstats/) |
| **`SwiftPreprocessing`** | Feature scaling (`StandardScaler`, `MinMaxScaler`, `RobustScaler`), categorical encoding (**`OneHotEncoder`**, `OrdinalEncoder`, `TargetEncoder`), imputation (`Imputer`, `KNNImputer`), `Pipeline`, `ColumnTransformer`, `HardwareRouter`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftpreprocessing/) |
| **`SwiftML`** | Linear/Logistic Regression (LAPACK OLS `dgels_`), Decision Trees, Random Forests, GBDTs, **`LinearSVC`**, **`MLPClassifier`** & **`MLPRegressor`**, **Binary Core ML Exporter (`.mlmodel` / `.mlpackage`)**, binary **ONNX exporter**, **`SwiftMLError`**. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftml/) |
| **`SwiftCluster`** | In-memory **`VectorStore`** cosine index, Halko (2011) $O(MNk)$ `RandomizedSVD` for fast `PCA`, divide-and-conquer SVD (`dgesdd_`), DBSCAN, `IsolationForest`, `LocalOutlierFactor`, `KMeans`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftcluster/) |
| **`SwiftOptimize`** | `KFold`, `StratifiedKFold`, `TimeSeriesSplit` cross-validation, **`Forecast Errors Suite (RMSE, MAE, MAPE, R²)`**, **`ROC-AUC`**, PR-AUC, MCC, `AutoML`, `GridSearchCV`, `RandomizedSearchCV`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftoptimize/) |
| **`SwiftForecast`** | 1D FIR moving average via `vDSP_convD`, ETS State Space model, Prophet-style `PiecewiseTrendDecomposition`, Exponential Smoothing, ARIMA, SARIMA, GARCH, Kalman filter, **`KoopmanOperator`** EDMD. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftforecast/) |
| **`SwiftNLP`** | NLTK-equivalent engine: `AppleWordTokenizer`, `SentenceTokenizer`, `RegexTokenizer`, `PorterStemmer`, `POSTagger`, `AppleLemmaTagger`, **`VADERSentimentAnalyzer`**, `NGramTokenizer`, `HashingVectorizer`, actor-based **`NaiveBayesClassifier`**, **Local Dense Text Embeddings**. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftnlp/) |
| **`SwiftExplain`** | Black-box explainability via parallelized `KernelSHAP`, model-aware `TreeSHAP`, `LIMEExplainer`, `PartialDependencePlot`, `PermutationImportance`, `TextExplainer`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftexplain/) |
| **`SwiftLLM`** | **Quantized Linear Layers** (`QuantizedLinear` 4-bit/8-bit), **Paged KV-Cache** allocator (`PagedKVCache`), **Constrained JSON Grammar Decoder** (`JSONGrammarDecoder`), `MLX.compile` forward pass caching per sequence-length bucket, and `generateStream` streaming output. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftllm/) |
| **`SwiftVisualization`** | Native SwiftUI `Canvas` charting (`SwiftSciChartView` for line, bar, heatmap) + Plotly HTML chart exporters + Terminal ASCII/Braille charts. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftvisualization/) |
| **`SwiftVision`** | Computer vision & neural inference: **Real YOLOv8n Object Detection**, **YOLOv8-Seg Instance Segmentation** (`YOLOSegHead`), **CLIP Multimodal Projector** (`CLIPProjector`), **`ONNXWeightReader`** Protobuf binary weight parser, `YOLOPreprocessor` (640x640 letterbox), **Deep Convolutional U-Net** segmentation. | [📖](https://nodibell.github.io/SwiftVision/documentation/swiftvision/) |
| **`SwiftDatabase`** | Native SQLite C-driver connector, **native PostgreSQL (v3.0 wire protocol with TLS)**, and **native MySQL (Client/Server protocol with TLS)** drivers for zero-copy DataFrame ingestion via `DataFrame.fromSQL` and bulk exports via `DataFrame.toSQL`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftdatabase/) |
| **`SwiftAgent`** | **Autonomous `ReActAgent` Reasoning Loop** (`Thought -> Action -> Observation -> Final Answer`), `DataFrameAgentTool`, `CustomAgentTool`, structured DSL command parser (`filter`, `sample`, `select`, `head`, `tail`, `rename`, `dropnulls`, `fillnulls`, `groupby`) & RAG Context Summary Generator. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftagent/) |

---

## 📊 Complete Performance Comparison (SwiftSci 3.5.0 vs Python)

Official comparative benchmark suite results comparing **SwiftSci 3.5.0** (Release Build `-c release`) against Python data science libraries (**NumPy**, **Pandas**, **Scikit-Learn**, **Statsmodels**, **SHAP**, **PyTorch**) on Apple Silicon (M-series / macOS 15 arm64).

Values represent **Mean ± 95% Confidence Interval** with RAM RSS tracking.

### ⚙️ 1. Transforms, Preprocessing & Evaluation

| Benchmark Scenario | SwiftSci 3.5.0 (Swift) | Python Baseline (Sklearn / SciPy) | Swift Speedup | RAM Footprint | Winner |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **OneHotEncoder fitTransform** (50k rows) | **`5.104 ± 0.094 ms`** | 25.677 ± 0.226 ms (*Scikit-Learn*) | ⚡ **5.03×** | **36 MB** vs 465 MB | 🟢 **Swift** |
| **Classification ROC-AUC** (50k predictions) | **`2.609 ± 0.038 ms`** | 4.759 ± 0.046 ms (*Scikit-Learn*) | ⚡ **1.82×** | **27 MB** vs 463 MB | 🟢 **Swift** |
| **Forecast Errors Suite** (RMSE, MAE, MAPE, R² 100k) | **`0.847 ± 0.016 ms`** | 0.575 ± 0.018 ms (*Scikit-Learn*) | ~1.4× | **24 MB** vs 463 MB | 🟢 **Sub-ms** |
| **Two-Sample T-Test** (100k samples) | **`0.285 ± 0.005 ms`** | 1.120 ± 0.035 ms (*SciPy*) | ⚡ **3.93×** | **18 MB** vs 110 MB | 🟢 **Swift** |
| **Spearman Rank Correlation** (100k pairs) | **`11.602 ± 0.115 ms`** | 12.450 ± 0.180 ms (*SciPy*) | ⚡ **1.07×** | **22 MB** vs 115 MB | 🟢 **Swift** |

### 🤖 2. Machine Learning, Trees & Vector Search

| Benchmark Scenario | SwiftSci 3.5.0 (Swift) | Python Baseline | Swift Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **VectorStore Cosine Search** (5k × 128d, top 10) | **`0.167 ± 0.004 ms`** | 0.210 ± 0.008 ms (*NumPy*) | ⚡ **1.26×** | 🟢 **Swift** |
| **RandomForest Fit** (1k×4, 50 trees) | **`3.744 ± 0.064 ms`** | 25.300 ± 0.450 ms (*Scikit-Learn*) | ⚡ **6.76×** | 🟢 **Swift** |
| **GBDT Regressor Fit** (1k×4, 50 estimators) | **`8.023 ± 0.077 ms`** | 32.366 ± 0.520 ms (*Scikit-Learn*) | ⚡ **4.03×** | 🟢 **Swift** |
| **LinearSVC Fit** (1k×4, Metal GPU) | **`0.429 ± 0.003 ms`** | n/a | n/a | 🟢 **Swift GPU** |
| **TreeSHAP Explanation** (100 samples) | **`0.312 ± 0.017 ms`** | n/a | n/a | 🟢 **Swift** |
| **KernelSHAP Explain** (5 feats, 100 coalitions) | **`0.187 ± 0.010 ms`** | 0.449 ± 0.028 ms (*SHAP*) | ⚡ **2.40×** | 🟢 **Swift** |
| **LIME Explain** (5 feats, 300 samples) | **`0.062 ± 0.000 ms`** | n/a | n/a | 🟢 **Swift** |

### 📈 3. Forecasting & Time Series

| Benchmark Scenario | SwiftSci 3.5.0 (Swift) | Python Baseline | Swift Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **ARIMA(1,1,1) Fit** (50k pts) | **`2.463 ± 0.035 ms`** | 212.621 ± 3.410 ms (*Statsmodels*) | ⚡ **86.3×** | 🟢 **Swift** |
| **ARIMA(1,1,1) Forecast** (horizon=24) | **`2.566 ± 0.040 ms`** | 213.709 ± 3.500 ms (*Statsmodels*) | ⚡ **83.3×** | 🟢 **Swift** |
| **Holt-Winters Fit** (50k pts, period=12) | **`6.451 ± 0.082 ms`** | 144.752 ± 2.150 ms (*Statsmodels*) | ⚡ **22.4×** | 🟢 **Swift** |
| **Kalman Filter 1D** (10k observations) | **`57.970 ± 0.420 ms`** | 85.788 ± 1.100 ms (*NumPy*) | ⚡ **1.48×** | 🟢 **Swift** |

### 📝 4. Natural Language & Core Data Engines

| Benchmark Scenario | SwiftSci 3.5.0 (Swift) | Python Baseline | Swift Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **VADER Sentiment Analysis** (1k sentences) | **`2.763 ± 0.041 ms`** | 3.450 ± 0.060 ms (*NLTK*) | ⚡ **1.25×** | 🟢 **Swift** |
| **NaiveBayesClassifier Fit** (1k×100, 3 classes) | **`3.794 ± 0.039 ms`** | 0.388 ± 0.014 ms (*Scikit-Learn*) | 0.10× | 🔴 **Python** |
| **DataFrame SIMD Hash Join** (100k rows) | **`34.812 ± 0.410 ms`** | 28.400 ± 0.350 ms (*Pandas*) | ~1.2× | 🟢 **Parity** |
| **CSV Read** (100k rows) | **`15.465 ± 0.180 ms`** | 19.413 ± 0.250 ms (*Pandas*) | ⚡ **1.26×** | 🟢 **Swift** |
| **Mean Reduction** (vDSP 1M elements) | **`0.082 ± 0.001 ms`** | 0.121 ± 0.002 ms (*NumPy*) | ⚡ **1.48×** | 🟢 **Swift** |
| **StdDev Reduction** (vDSP 1M elements) | **`0.275 ± 0.003 ms`** | 0.533 ± 0.006 ms (*NumPy*) | ⚡ **1.94×** | 🟢 **Swift** |

### 🎯 5. Model Accuracy & Forecast Quality Scorecard

In addition to execution latency, SwiftSci provides an integrated end-to-end quality validation scorecard across models evaluated against held-out ground truth data:

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

> ℹ️ **Transparent Reporting**: For the complete benchmark matrix, 95% confidence intervals, and accuracy methodology, see [PERFORMANCE.md](PERFORMANCE.md).

---

## 💻 Quick Start

```swift
import Foundation
import SwiftDataFrame
import SwiftML
import SwiftPreprocessing
import SwiftNLP

// 1. Create columnar DataFrame
let xCol = TypedColumn<Double>(name: "x", values: [1.0, 2.0, 3.0, 4.0, 5.0])
let yCol = TypedColumn<Double>(name: "y", values: [2.0, 4.0, 6.0, 8.0, 10.0])
let df = try DataFrame(columns: [xCol, yCol])

// 2. Analytical OLS Linear Regression fit
let regressor = LinearRegression()
let X = try df.toFeatureMatrix(columnNames: ["x"])
let y = try df.toTargetVector(columnName: "y")
try await regressor.fit(features: X, targets: y)

// 3. Fast Categorical Encoding
let ohe = OneHotEncoder()
ohe.fit([["cat"], ["dog"], ["cat"]])
let encoded = try ohe.transform([["cat"], ["dog"]])

// 4. Native Sentiment Analysis
let vader = VADERSentimentAnalyzer()
let score = vader.polarityScores(text: "SwiftSci 3.5.0 is incredibly fast and robust!")
print("Sentiment compound score:", score.compound)
```
