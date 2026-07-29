# SwiftSci 2.2

**SwiftSci** is a native, high-performance, modular data analysis and machine learning library for Swift. It is built from the ground up to leverage Apple Silicon (M-series) unified memory architecture (UMA) and is fully compliant with Swift 6 strict concurrency requirements.

The package combines hardware-accelerated tensor computations on the Apple Silicon GPU via **MLX** with highly optimized CPU vector routines from the **Accelerate framework (vDSP / LAPACK / BLAS)**.

[![Swift Version](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/Nodibell/SwiftSci)
[![Platform Compatibility](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/Nodibell/SwiftSci)
[![Documentation](https://img.shields.io/badge/docs-DocC-blue)](https://nodibell.github.io/SwiftSci/)


---

## 🚀 Core Modules (14 Targets)

| Module | Description | Docs |
| :--- | :--- | :---: |
| **`SwiftDataFrame`** | Columnar data frame with Arrow zero-copy semantics, streaming CSV/JSON, `DataFrame.readURL` HTTP ingestion, hash joins, `pivot`/`melt`, `mapColumn`, `parallelGathered(at:)`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftdataframe/) |
| **`SwiftStats`** | Vectorized descriptive statistics, SIMD vDSP sorting, Student-t/Chi-Square/F distributions, paired t-test, ANOVA powered by `Accelerate vDSP`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftstats/) |
| **`SwiftPreprocessing`** | Feature scaling, categorical encoding (`OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`, `FrequencyEncoder`), imputation (`Imputer`, `KNNImputer`), `Pipeline`, `ColumnTransformer`, `HardwareRouter`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftpreprocessing/) |
| **`SwiftML`** | Linear/Logistic Regression (with LAPACK OLS `dgels_`), Decision Trees (pre-sorted feature splits), Random Forests, GBDTs, `OneVsRestClassifier`, `MLPClassifier`/`MLPRegressor` with Adam optimizer & BLAS `cblas_dgemm`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftml/) |
| **`SwiftCluster`** | Divide-and-conquer SVD PCA (`dgesdd_`), DBSCAN, `IsolationForest`, `LocalOutlierFactor`, `KMeans` with KMeans++ init. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftcluster/) |
| **`SwiftOptimize`** | `KFold`, `StratifiedKFold`, `TimeSeriesSplit` cross-validation, ROC-AUC, PR-AUC, MCC, `GridSearchCV`, `RandomizedSearchCV`, `AutoML` search engine. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftoptimize/) |
| **`SwiftForecast`** | Exponential Smoothing, ARIMA, SARIMA, GARCH, Kalman filter, `ExpandingWindow`, additive/multiplicative decomposition. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftforecast/) |
| **`SwiftNLP`** | `TextNormalizer`, Ukrainian stopwords, BPE tokenizer, `NGramTokenizer`, `HashingVectorizer`, `TFIDFVectorizer`, embeddings. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftnlp/) |
| **`SwiftExplain`** | Black-box explainability via parallelized `KernelSHAP`, model-aware `TreeSHAP`, `PartialDependencePlot`, `PermutationImportance`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftexplain/) |
| **`SwiftLLM`** | Local GPU text generation via causal transformer-decoder. SafeTensors & GGUF weight parsing, Top-K/Top-P sampling. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftllm/) |
| **`SwiftVisualization`** | Standalone interactive Plotly HTML chart exporters with dynamic ROC & trapezoidal AUC score: `plotCorrelationHeatmap`, `plotROCCurve`, `plotFeatureImportances`, `plotConfusionMatrix`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftvisualization/) |
| **`SwiftVision`** | Computer vision pipeline: `ImageDataset`, U-Net Segmentation, object detection wrappers, CNN feature extraction, `VisionError`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftvision/) |
| **`SwiftDatabase`** | Native SQLite C-driver connector (`sqlite3_open_v2`) for zero-copy DataFrame ingestion via `DataFrame.fromSQL`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftdatabase/) |
| **`SwiftAgent`** | Structured DSL command parser & RAG Context Summary Generator for local LLMs. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftagent/) |

---

## 📊 Complete Performance Comparison (SwiftSci 2.2 vs Python)

The following table presents median execution times for benchmark scenarios on Apple Silicon (M-series / macOS 15 arm64), compared directly against popular Python counterparts (**Scikit-Learn, NumPy, Pandas, SHAP, Statsmodels, PyTorch**). See [PERFORMANCE.md](PERFORMANCE.md) for full benchmark methodology.

### 📈 1. Time Series Forecasting & Volatility
| Benchmark Scenario | SwiftSci 2.2 (Swift) | Python Baseline | Swift Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **ARIMA(1,1,1) Fit** (50k pts) | **2.48 ms** | 227.34 ms (*Statsmodels*) | ⚡ **91.6×** | 🟢 Swift |
| **ARIMA(1,1,1) Forecast** (horizon=24) | **2.49 ms** | 224.57 ms (*Statsmodels*) | ⚡ **90.3×** | 🟢 Swift |
| **Holt-Winters Fit** (50k pts, period=12) | **7.42 ms** | 144.90 ms (*Statsmodels*) | ⚡ **19.5×** | 🟢 Swift |

### 🤖 2. Machine Learning & Clustering
| Benchmark Scenario | SwiftSci 2.2 (Swift) | Python Baseline | Swift Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **RandomForest Fit** (1k×4, 50 trees) | **4.63 ms** | 27.10 ms (*Scikit-Learn*) | ⚡ **5.86×** | 🟢 Swift |
| **GBDT Regressor Fit** (1k×4, 50 est) | **8.51 ms** | 34.80 ms (*Scikit-Learn*) | ⚡ **4.09×** | 🟢 Swift |

### 📝 3. Natural Language & Explainability
| Benchmark Scenario | SwiftSci 2.2 (Swift) | Python Baseline | Swift Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **KernelSHAP Explain** (100 coalitions) | **0.18 ms** | 0.46 ms (*SHAP*) | ⚡ **2.57×** | 🟢 Swift |
| **LLM Forward Pass** (seqLen=64) | **0.51 ms** | 0.67 ms (*PyTorch*) | ⚡ **1.31×** | 🟢 Swift |

### 📊 4. Core Data Engines & Vector Stats
| Benchmark Scenario | SwiftSci 2.2 (Swift) | Python Baseline | Swift Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Reduction** (vDSP 1M elements) | **0.082 ms** | 0.118 ms (*NumPy*) | ⚡ **1.44×** | 🟢 Swift |
| **StdDev Reduction** (vDSP 1M elements) | **0.311 ms** | 0.516 ms (*NumPy*) | ⚡ **1.66×** | 🟢 Swift |
| **Pearson Correlation** (500k pairs) | **0.866 ms** | 1.233 ms (*NumPy*) | ⚡ **1.42×** | 🟢 Swift |
| **CSV Read** (100k rows) | **16.53 ms** | 20.11 ms (*Pandas*) | ⚡ **1.22×** | 🟢 Swift |
| **CSV Stream + GroupBy** (100k rows) | **22.88 ms** | 30.05 ms (*Pandas*) | ⚡ **1.31×** | 🟢 Swift |

> ℹ️ **Transparent Reporting**: For the complete 25-benchmark matrix (including Python wins such as C-indexed DataFrame row filtering, matrix sorting, and analysis of performance gaps tracked for v2.3), see [PERFORMANCE.md](PERFORMANCE.md).

---

## 🛠 Architectural Highlights & Optimizations

### 1. Data-Oriented Design (DOD) Tree Ensembles
Traditional object-oriented trees (where every node is a reference type containing child node pointers) suffer from severe Automatic Reference Counting (ARC) overhead and poor CPU cache locality. `SwiftSci` stores trees as contiguous flat arrays of `FlatTreeNode` structures with pre-sorted feature matrix split indexes, eliminating pointer chasing and maximizing L1/L2 cache hit rates.

### 2. Intelligent Hardware Routing
We implement a flexible compute device routing policy (`requestedDevice` / `resolvedDevice`):
* Branch-heavy algorithms (Decision Trees, Random Forests, spatial DBSCAN search) execute on CPU vector cores.
* Tensor operations (Linear/Logistic Regression, Multi-Layer Perceptrons) leverage Apple Silicon GPU execution via `MLXArray` lazy evaluation and Accelerate `BLAS` (`cblas_dgemm`).

### 3. Direct Remote Dataset Ingestion & Native SQL
`SwiftDataFrame` supports streaming and parsing datasets directly from remote HTTP/HTTPS URLs into zero-copy DataFrames using `DataFrame.readURL(url)`, as well as direct native C-driver SQLite query execution via `SQLiteConnection`.

---

## 💻 Quick Start

```swift
import Foundation
import SwiftDataFrame
import SwiftML

// 1. Create columnar DataFrame
let xCol = TypedColumn<Double>(name: "x", values: [1.0, 2.0, 3.0, 4.0, 5.0])
let yCol = TypedColumn<Double>(name: "y", values: [2.0, 4.0, 6.0, 8.0, 10.0])
let df = try DataFrame(columns: [xCol, yCol])

// 2. Analytical OLS Linear Regression fit
let regressor = LinearRegression()
let X = try df.toFeatureMatrix(columnNames: ["x"])
let y = try df.toTargetVector(columnName: "y")
try regressor.fit(features: X, target: y)

print("Coefficients:", regressor.weights)
```
