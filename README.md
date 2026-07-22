# SwiftSci

**SwiftSci** is a native, high-performance, modular data analysis and machine learning library for Swift. It is built from the ground up to leverage Apple Silicon (M-series) unified memory architecture (UMA) and is fully compliant with Swift 6 strict concurrency requirements.

The package combines hardware-accelerated tensor computations on the Apple Silicon GPU via **MLX** with highly optimized CPU vector routines from the **Accelerate framework (vDSP / LAPACK)**.

[![Swift Version](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/Nodibell/SwiftSci)
[![Platform Compatibility](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/Nodibell/SwiftSci)

---

## 🚀 Core Modules

## 🚀 Core Modules

* **`SwiftDataFrame`**: High-performance columnar data manipulation with zero-copy semantics, built on top of `Apache Arrow`. Features `SystemsCSVParser` (zero-copy memory-mapped RFC 4180 DFA byte scanner) and parallel chunk-buffered streaming.
* **`SwiftStats`**: Vectorized descriptive statistics, distributions, and hypothesis tests powered by `Accelerate vDSP`.
* **`SwiftPreprocessing`**: Feature scaling, categorical encoding, binning, feature selection (`SelectKBest`, `VarianceThreshold`, `RecursiveFeatureElimination` RFE), and preprocessing pipelines (`Pipeline`).
* **`SwiftML`**: Classical estimators (Linear/Logistic Regression, Decision Trees, Random Forests, GBDTs) with Gini `featureImportances` and `Codable` model persistence (`save`/`load`).
* **`SwiftCluster`**: Dimensionality reduction (SVD-based PCA, DBSCAN), outlier detection (`IsolationForest`, `LocalOutlierFactor`), and clustering (`KMeans` with KMeans++ init).
* **`SwiftOptimize`**: Model validation (`KFold` cross-validation), evaluation metrics, and parallel hyperparameter optimization (`GridSearchCV`).
* **`SwiftForecast`**: Time series analysis (decomposition, Holt-Winters, ARIMA, SARIMA seasonal models, GARCH volatility, Kalman filtering, `ExpandingWindow`).
* **`SwiftNLP`**: Tokenization (Word, subword BPE, `NGramTokenizer`), feature extraction (`HashingVectorizer`, `CountVectorizer`, `TFIDFVectorizer`), and static word embeddings.
* **`SwiftExplain`**: Black-box model explainability using a parallelized `KernelSHAP` implementation.
* **`SwiftLLM`**: Local text generation on GPU using causal transformer-decoder architectures and MLX. Supports SafeTensors and GGUF weight parsing.

---

## 📊 Performance Comparison (Swift vs Python)

The following table presents median execution times on an **Apple Silicon M-series (macOS 15 / arm64)** system, compared directly against popular Python counterparts (Scikit-Learn, NumPy, SHAP, Statsmodels, PyTorch):

| Benchmark Test | Swift (ms) | Python (ms) | Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Mean** (1M elements) | 0.081 ms | 0.123 ms | 1.52x | 🟢 Swift |
| **StdDev** (1M elements) | 0.452 ms | 0.499 ms | 1.10x | 🟢 Swift |
| **Variance** (1M elements) | 0.476 ms | 0.500 ms | 1.05x | 🟢 Swift |
| **Pearson Correlation** (500k elements) | 0.765 ms | 1.159 ms | 1.52x | 🟢 Swift |
| **CSV Read** (100k rows, 5 cols) | 69.143 ms | 18.053 ms | 0.26x | 🔴 Python |
| **CSV Stream Read** (chunk=10k) | 239.801 ms | 20.368 ms | 0.08x | 🔴 Python |
| **CSV Stream + Filter** | 244.250 ms | 22.597 ms | 0.09x | 🔴 Python |
| **CSV Stream + GroupBy** | 244.690 ms | 25.488 ms | 0.10x | 🔴 Python |
| **Filter rows** (100k rows) | 29.606 ms | 0.537 ms | 0.02x | 🔴 Python |
| **GroupBy + sum/mean** (4 groups) | 2.324 ms | 1.593 ms | 0.69x | 🔴 Python |
| **SortBy double column** (100k rows) | 76.556 ms | 6.506 ms | 0.08x | 🔴 Python |
| **LinearRegression fit** (10k×10, 100 epochs) | 27.538 ms | 23.542 ms | 0.85x | 🔴 Python |
| **Random Forest fit** (1k x 4 features, 50 trees) | 4.781 ms | 24.030 ms | 5.03x | 🟢 Swift |
| **GBDT Regressor fit** (1k x 4, 50 estimators) | 32.803 ms | 30.839 ms | 0.94x | 🔴 Python |
| **KMeans fit** (10k x 4, 3 clusters) | 50.827 ms | 11.877 ms | 0.23x | 🔴 Python |
| **PCA SVD fit** (1k×100 → 10 components) | 2.024 ms | 0.787 ms | 0.39x | 🔴 Python |
| **Holt-Winters fit** (50k points, period=12) | 6.310 ms | 133.764 ms | 21.20x | 🟢 Swift |
| **ARIMA fit** (50k points, Hannan-Rissanen) | 2.274 ms | 201.623 ms | 88.68x | 🟢 Swift |
| **ARIMA forecast horizon=24** (50k points) | 2.322 ms | 202.574 ms | 87.23x | 🟢 Swift |
| **Kalman Filter 1D** (10k observations) | 58.323 ms | 80.455 ms | 1.38x | 🟢 Swift |
| **TS Decomposition additive** (1k points) | 0.437 ms | 0.093 ms | 0.21x | 🔴 Python |
| **LLM Forward Pass** (seqLen=64) | 0.739 ms | 0.457 ms | 0.62x | 🔴 Python |
| **LLM Generate** (10 tokens) | 5.394 ms | 3.540 ms | 0.66x | 🔴 Python |
| **KernelSHAP Explain** (5 features, 100 coalitions) | 0.176 ms | 0.426 ms | 2.42x | 🟢 Swift |

---

## 🛠 Architectural Highlights & Optimizations

### 1. Transitioning from OOP to Data-Oriented Design (DOD)
Traditional object-oriented trees (where every node is a reference type containing child node pointers) suffer from severe Automatic Reference Counting (ARC) overhead and poor CPU cache locality (L1/L2 cache misses). 
`SwiftSci` resolves this by storing trees as a contiguous flat array of `FlatTreeNode` structures:
* Nodes are allocated next to each other in memory.
* Child navigation is done via array offsets.
* This dramatically increases tree ensemble traversal speeds.

### 2. Intelligent Hardware Routing
We implement a flexible compute device routing policy (`requestedDevice` / `resolvedDevice`):
* Branch-heavy algorithms (such as Decision Trees, Random Forests, or spatial DBSCAN search) run strictly on CPU.
* Large tensor operations (Linear/Logistic Regression, K-Means) leverage Apple Silicon GPU execution blocks via `MLXArray` lazy evaluation.

### 3. Optional NaN Validation Bypass
We identified that running $O(N)$ CPU sweeps to check for `NaN` presence in statistical calculations adds significant CPU bottlenecks that limit the speed of underlying `vDSP` functions. In v1.0, we introduced a `checkNaN: Bool = true` default parameter to descriptive stats. When bypassed (set to `false` in benchmarks and hot loops), SwiftStats leverages pure hardware SIMD speeds, outperforming NumPy.

---

## ⚠️ Known Gaps & Limitations

1. **Standard CSV Parser Overhead**: The standard CSV reader runs with $O(N)$ CPU overhead. In v1.1.0, this is resolved by using the parallel streaming CSV parser (`DataFrame.readCSVStream`) returning `AsyncThrowingStream` for high-performance file parsing.
2. **MLX Sendable Isolation**: `MLXArray` does not conform to `Sendable`. This is handled in the library through strict actor isolation and passing memory ownership tokens (`WiredMemoryTicket`).

---

## 💻 Quick Start

```swift
import SwiftDataFrame
import SwiftStats
import SwiftML

// 1. Load CSV data
let df = try DataFrame.readCSV(contentsOf: csvURL)

// 2. Compute descriptive statistics without NaN checks for maximum speed
let summary = try Stats.describe(df["target"].toDoubles()!, checkNaN: false)
print(summary)

// 3. Train a GPU-accelerated Estimator
let regressor = LinearRegression(device: .gpu)
try await regressor.fit(features: X, targets: y)
let predictions = try await regressor.predict(features: X_test)
```

---

## 🗺 Roadmap & Future Plans

For detailed implementation plans and ecosystem roadmap, see the [ROADMAP](file:///Users/oleksiichumak/Developer/Xcode.projects/SwiftAnalytics/ROADMAP/ROADMAP.md) directory:
* **v1.1 (Completed 🟢)**: Streaming CSV Parser, SafeTensors & GGUF model loader, SARIMA & GARCH models.
* **v1.2 (Completed 🟢)**: Package renaming (`SwiftSci`), Kalman filter Joseph form fix, byte-level BPE, `addColumn`.
* **v1.3 (Planned 📋)**: Sklearn parity (`ClassificationPipeline`/`RegressionPipeline`, `ColumnTransformer`, `RandomizedSearchCV`, extended metrics, outlier detection).

---

## 📜 License
This project is licensed under the MIT License.
