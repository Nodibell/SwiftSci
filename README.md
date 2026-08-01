# SwiftSci 2.5.0

**SwiftSci** is a native, high-performance, modular data analysis and machine learning library for Swift. It is built from the ground up to leverage Apple Silicon (M-series) unified memory architecture (UMA) and is fully compliant with Swift 6 strict concurrency requirements.

The package combines hardware-accelerated tensor computations on the Apple Silicon GPU via **MLX** with highly optimized CPU vector routines from the **Accelerate framework (vDSP / LAPACK / BLAS)** and Apple's native **NaturalLanguage** framework.

[![Swift Version](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/Nodibell/SwiftSci)
[![Platform Compatibility](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/Nodibell/SwiftSci)
[![Documentation](https://img.shields.io/badge/docs-DocC-blue)](https://nodibell.github.io/SwiftSci/)

---

## 🌐 Platform Support & SPI Compatibility

SwiftSci is engineered for multi-platform deployment across Apple Silicon ecosystems:

* **macOS 14+ (Apple Silicon M-Series)**: Supports all **14 core modules**, leveraging Accelerate (vDSP/LAPACK/BLAS) and MLX Metal GPU acceleration (`SwiftPreprocessing`, `SwiftML`, `SwiftCluster`, `SwiftLLM`, `SwiftExplain`, `SwiftVision`, `SwiftAgent`).
* **iOS 18+ & visionOS 2+**: Supports all pure CPU vector analytics modules (`SwiftDataFrame`, `SwiftStats`, `SwiftNLP`, `SwiftForecast`, `SwiftVisualization`, `SwiftDatabase`). MLX-dependent targets are conditionally built on macOS via `.when(platforms: [.macOS])`.

---

## 🚀 Core Modules & What's New in 2.5.0

| Module                           | Description                                                                                                                                                                                                                                                                                                                                                          |                                    Docs                                    |
| :------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------: |
| **`SwiftDataFrame`**     | Columnar data frame with Arrow zero-copy semantics, Feather / Arrow IPC serialization (`FeatherReader`/`FeatherWriter`), deferred `LazyDataFrame` with filter predicate merging, streaming CSV/JSON, `pivot`/`melt`, `mapColumn`. |   [📖](https://nodibell.github.io/SwiftSci/documentation/swiftdataframe/)   |
| **`SwiftStats`**         | Vectorized descriptive statistics, SIMD vDSP sorting, Student-t/Chi-Square/F distributions, paired t-test, ANOVA powered by `Accelerate vDSP`.                                                                                                                                                                                                                      |     [📖](https://nodibell.github.io/SwiftSci/documentation/swiftstats/)     |
| **`SwiftPreprocessing`** | Feature scaling, categorical encoding (`OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`, `FrequencyEncoder`), imputation (`Imputer`, `KNNImputer`), `Pipeline`, `ColumnTransformer`, `HardwareRouter`.                                                                                                                                           | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftpreprocessing/) |
| **`SwiftML`**            | Linear/Logistic Regression (LAPACK OLS `dgels_`), Decision Trees, Random Forests, GBDTs, parallel `MultiOutputRegressor` & `MultiLabelClassifier`, `MLPClassifier`/`MLPRegressor` with Adam optimizer & BLAS `cblas_dgemm`. |      [📖](https://nodibell.github.io/SwiftSci/documentation/swiftml/)      |
| **`SwiftCluster`**       | Divide-and-conquer SVD PCA (`dgesdd_`), DBSCAN, `IsolationForest`, `LocalOutlierFactor`, `KMeans` with KMeans++ init.                                                                                                                                                                                                                                        |    [📖](https://nodibell.github.io/SwiftSci/documentation/swiftcluster/)    |
| **`SwiftOptimize`**      | `KFold`, `StratifiedKFold`, `TimeSeriesSplit` cross-validation, ROC-AUC, PR-AUC, MCC, `GridSearchCV`, generalized `RandomizedSearchCV` (`searchGeneric`).                                                                                                                                                                                                      |   [📖](https://nodibell.github.io/SwiftSci/documentation/swiftoptimize/)   |
| **`SwiftForecast`**      | ETS State Space model (`autoFit` AICc selection), Prophet-style `PiecewiseTrendDecomposition`, Exponential Smoothing, ARIMA, SARIMA, GARCH, Kalman filter. |   [📖](https://nodibell.github.io/SwiftSci/documentation/swiftforecast/)   |
| **`SwiftNLP`**           | NLTK-equivalent engine: `AppleWordTokenizer`, `SentenceTokenizer`, `RegexTokenizer`, `PorterStemmer`, `POSTagger`, `AppleLemmaTagger`, `VADERSentimentAnalyzer`, `MultinomialNaiveBayes`, `df.vectorizeTextColumn`. |      [📖](https://nodibell.github.io/SwiftSci/documentation/swiftnlp/)      |
| **`SwiftExplain`**       | Black-box explainability via parallelized `KernelSHAP`, model-aware `TreeSHAP`, `PartialDependencePlot`, `PermutationImportance`, `TextExplainer`.                                                                                                                                                                                                          |    [📖](https://nodibell.github.io/SwiftSci/documentation/swiftexplain/)    |
| **`SwiftLLM`**           | Local GPU text generation with `KVCache` Key-Value tensor cache and `generateStream` `AsyncThrowingStream` streaming output. SafeTensors & GGUF weight parsing, Top-K/Top-P sampling. |      [📖](https://nodibell.github.io/SwiftSci/documentation/swiftllm/)      |
| **`SwiftVisualization`** | Native SwiftUI `Canvas` charting (`SwiftSciChartView` for line, bar, heatmap) + Plotly HTML chart exporters. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftvisualization/) |
| **`SwiftVision`**        | Computer vision pipeline: `ImageDataset`, U-Net Segmentation, object detection wrappers, CNN feature extraction. |    [📖](https://nodibell.github.io/SwiftSci/documentation/swiftvision/)    |
| **`SwiftDatabase`**      | Native SQLite C-driver connector (`sqlite3_open_v2`) for zero-copy DataFrame ingestion via `DataFrame.fromSQL`. |   [📖](https://nodibell.github.io/SwiftSci/documentation/swiftdatabase/)   |
| **`SwiftAgent`**         | Structured DSL command parser & RAG Context Summary Generator for local LLMs.                                                                                                                                                                                                                                                                                        |     [📖](https://nodibell.github.io/SwiftSci/documentation/swiftagent/)     |
| **`SwiftSciCLI`**        | Command-line interface (`swiftsci summary`, `swiftsci convert`, `swiftsci export-model`). | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftscicli/) |


---

## 📊 Complete Performance Comparison (SwiftSci 2.5 vs Python)

The following table presents median execution times for benchmark scenarios on Apple Silicon (M-series / macOS 15 arm64), compared directly against popular Python counterparts (**Scikit-Learn, NumPy, Pandas, SHAP, Statsmodels, PyTorch**). Benchmarks are executed using Release mode optimizations (`swift run -c release`). See [PERFORMANCE.md](PERFORMANCE.md) for the complete 25-benchmark matrix.

### 📈 1. Time Series Forecasting & Volatility

| Benchmark Scenario                              | SwiftSci 2.5 (Swift) |       Python Baseline       |    Swift Speedup    |  Winner  |
| :---------------------------------------------- | :------------------: | :-------------------------: | :-----------------: | :------: |
| **ARIMA(1,1,1) Fit** (50k pts)            |  **2.41 ms**  | 227.34 ms (*Statsmodels*) | ⚡**94.4×** | 🟢 Swift |
| **ARIMA(1,1,1) Forecast** (horizon=24)    |  **2.58 ms**  | 224.57 ms (*Statsmodels*) | ⚡**87.1×** | 🟢 Swift |
| **Holt-Winters Fit** (50k pts, period=12) |  **7.16 ms**  | 144.90 ms (*Statsmodels*) | ⚡**20.2×** | 🟢 Swift |

### 🤖 2. Machine Learning & Clustering

| Benchmark Scenario                           | SwiftSci 2.5 (Swift) |       Python Baseline       |   Swift Speedup   |  Winner  |
| :------------------------------------------- | :------------------: | :-------------------------: | :----------------: | :------: |
| **RandomForest Fit** (1k×4, 50 trees) |  **3.94 ms**  | 27.10 ms (*Scikit-Learn*) | ⚡**6.87×** | 🟢 Swift |
| **GBDT Regressor Fit** (1k×4, 50 est) |  **8.61 ms**  | 34.80 ms (*Scikit-Learn*) | ⚡**4.04×** | 🟢 Swift |

### 📝 3. Natural Language & Explainability

| Benchmark Scenario                            | SwiftSci 2.5 (Swift) |    Python Baseline    |   Swift Speedup   |  Winner  |
| :-------------------------------------------- | :------------------: | :-------------------: | :----------------: | :------: |
| **KernelSHAP Explain** (100 coalitions) |  **0.19 ms**  |  0.46 ms (*SHAP*)  | ⚡**2.42×** | 🟢 Swift |
| **LLM Forward Pass** (seqLen=64)        |  **0.65 ms**  | 0.67 ms (*PyTorch*) | ⚡**1.03×** | 🟢 Swift |

### 📊 4. Core Data Engines & Vector Stats

| Benchmark Scenario                            | SwiftSci 2.5 (Swift) |    Python Baseline    |   Swift Speedup   |  Winner  |
| :-------------------------------------------- | :------------------: | :-------------------: | :----------------: | :------: |
| **Mean Reduction** (vDSP 1M elements)   |  **0.083 ms**  | 0.118 ms (*NumPy*) | ⚡**1.42×** | 🟢 Swift |
| **StdDev Reduction** (vDSP 1M elements) |  **0.458 ms**  | 0.516 ms (*NumPy*) | ⚡**1.13×** | 🟢 Swift |
| **Pearson Correlation** (500k pairs)    |  **1.037 ms**  | 1.233 ms (*NumPy*) | ⚡**1.19×** | 🟢 Swift |
| **Kalman Filter 1D** (10k obs)          |  **65.03 ms**  | 87.78 ms (*NumPy*) | ⚡**1.35×** | 🟢 Swift |
| **CSV Read** (100k rows)                |  **16.45 ms**  | 20.11 ms (*Pandas*) | ⚡**1.22×** | 🟢 Swift |
| **CSV Stream + GroupBy** (100k rows)    |  **22.28 ms**  | 30.05 ms (*Pandas*) | ⚡**1.35×** | 🟢 Swift |

> ℹ️ **Transparent Reporting**: For the complete 25-benchmark matrix, see [PERFORMANCE.md](PERFORMANCE.md).


---

## 🛠 Architectural Highlights & Optimizations

### 1. Data-Oriented Design (DOD) Tree Ensembles

Traditional object-oriented trees suffer from ARC overhead and cache misses. `SwiftSci` stores trees as contiguous flat arrays of `FlatTreeNode` structures with pre-sorted feature matrix split indexes, eliminating pointer chasing and maximizing CPU L1/L2 cache hit rates.

### 2. Intelligent Hardware Routing

We implement a flexible compute device routing policy (`requestedDevice` / `resolvedDevice`):

* Branch-heavy algorithms (Decision Trees, Random Forests, spatial DBSCAN search) execute on CPU vector cores.
* Tensor operations (Linear/Logistic Regression, Multi-Layer Perceptrons) leverage Apple Silicon GPU execution via `MLXArray` lazy evaluation and Accelerate `BLAS` (`cblas_dgemm`).

### 3. Native NaturalLanguage Integration & SwiftNLP

`SwiftNLP` bridges NLTK-equivalent algorithms (`PorterStemmer`, `VADERSentimentAnalyzer`, `MultinomialNaiveBayes`) with native Apple OS frameworks (`NLTagger`, `NLLanguageRecognizer`, `NLEmbedding`), ensuring high accuracy and zero external binary dependencies.

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
let score = vader.polarityScores(text: "SwiftSci 2.4.0 is super fast!")
print("Sentiment compound score:", score.compound)
```
