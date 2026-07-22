# SwiftAnalytics

**SwiftAnalytics** is a native, high-performance, modular data analysis and machine learning library for Swift. It is built from the ground up to leverage Apple Silicon (M-series) unified memory architecture (UMA) and is fully compliant with Swift 6 strict concurrency requirements.

The package combines hardware-accelerated tensor computations on the Apple Silicon GPU via **MLX** with highly optimized CPU vector routines from the **Accelerate framework (vDSP / LAPACK)**.

---

## 🚀 Core Modules

* **`SwiftDataFrame`**: High-performance columnar data manipulation with zero-copy semantics, built on top of `Apache Arrow`. Now supports parallel chunk-buffered streaming CSV parsing.
* **`SwiftStats`**: Vectorized descriptive statistics, distributions, and hypothesis tests powered by `Accelerate vDSP`.
* **`SwiftPreprocessing`**: Feature scaling, categorical encoding, binning, and preprocessing pipelines (`Pipeline`).
* **`SwiftML`**: Classical estimators (Linear/Logistic Regression, Decision Trees, Random Forests, GBDTs).
* **`SwiftCluster`**: Dimensionality reduction (SVD-based PCA, DBSCAN) and clustering (K-Means).
* **`SwiftOptimize`**: Model validation (K-Fold cross-validation) and parallel hyperparameter optimization (`GridSearchCV`).
* **`SwiftForecast`**: Time series analysis (decomposition, Holt-Winters, ARIMA, SARIMA seasonal models, GARCH volatility, Kalman filtering).
* **`SwiftNLP`**: Tokenization (Word / subword BPE Tokenizers) and static word embeddings.
* **`SwiftExplain`**: Black-box model explainability using a parallelized `KernelSHAP` implementation.
* **`SwiftLLM`**: Local text generation on GPU using casual transformer-decoder architectures and MLX. Supports zero-copy SafeTensors and GGUF weight parsing.

---

## 📊 Performance Comparison (Swift vs Python)

The following table presents median execution times on an **Apple Silicon M-series (macOS 15 / arm64)** system, compared directly against popular Python counterparts (Scikit-Learn, NumPy, SHAP, Statsmodels, PyTorch):

| Benchmark Test | Swift (ms) | Python (ms) | Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Mean** (1M elements) | 0.081 ms | 0.119 ms | 1.47x | 🟢 Swift |
| **StdDev** (1M elements) | 0.486 ms | 0.502 ms | 1.03x | 🟢 Swift |
| **Variance** (1M elements) | 0.467 ms | 0.506 ms | 1.08x | 🟢 Swift |
| **Pearson Correlation** (500k elements) | 0.790 ms | 1.165 ms | 1.47x | 🟢 Swift |
| **CSV Read** (100k rows, 5 cols) | 163.315 ms | 18.148 ms | 0.11x | 🔴 Python |
| **CSV Stream Read** (chunk=10k) | 238.009 ms | 20.444 ms | 0.09x | 🔴 Python |
| **CSV Stream + Filter** | 240.945 ms | 22.460 ms | 0.09x | 🔴 Python |
| **CSV Stream + GroupBy** | 230.476 ms | 25.282 ms | 0.11x | 🔴 Python |
| **Filter rows** (100k rows) | 30.608 ms | 0.537 ms | 0.02x | 🔴 Python |
| **GroupBy + sum/mean** (4 groups) | 2.375 ms | 1.580 ms | 0.67x | 🔴 Python |
| **SortBy double column** (100k rows) | 77.727 ms | 6.434 ms | 0.08x | 🔴 Python |
| **LinearRegression fit** (10k×10, 100 epochs) | 24.914 ms | 24.228 ms | 0.97x | 🔴 Python |
| **Random Forest fit** (1k x 4 features, 50 trees) | 4.440 ms | 24.371 ms | 5.49x | 🟢 Swift |
| **GBDT Regressor fit** (1k x 4, 50 estimators) | 34.072 ms | 29.810 ms | 0.87x | 🔴 Python |
| **KMeans fit** (10k x 4, 3 clusters) | 39.499 ms | 11.548 ms | 0.29x | 🔴 Python |
| **PCA SVD fit** (1k×100 → 10 components) | 1.977 ms | 0.733 ms | 0.37x | 🔴 Python |
| **Holt-Winters fit** (50k points, period=12) | 6.454 ms | 136.864 ms | 21.21x | 🟢 Swift |
| **ARIMA fit** (50k points) | 2.229 ms | 204.534 ms | 91.76x | 🟢 Swift |
| **ARIMA forecast horizon=24** (50k points) | 2.293 ms | 203.017 ms | 88.52x | 🟢 Swift |
| **Kalman Filter 1D** (10k observations) | 45.275 ms | 81.305 ms | 1.80x | 🟢 Swift |
| **TS Decomposition additive** (1k points) | 0.453 ms | 0.093 ms | 0.20x | 🔴 Python |
| **LLM Forward Pass** (seqLen=64) | 0.449 ms | 0.768 ms | 1.71x | 🟢 Swift |
| **LLM Generate** (10 tokens) | 4.283 ms | 4.377 ms | 1.02x | 🟢 Swift |
| **KernelSHAP Explain** (5 features, 100 coalitions) | 0.208 ms | 0.450 ms | 2.16x | 🟢 Swift |

---

## 🛠 Architectural Highlights & Optimizations

### 1. Transitioning from OOP to Data-Oriented Design (DOD)
Traditional object-oriented trees (where every node is a reference type containing child node pointers) suffer from severe Automatic Reference Counting (ARC) overhead and poor CPU cache locality (L1/L2 cache misses). 
`SwiftAnalytics` resolves this by storing trees as a contiguous flat array of `FlatTreeNode` structures:
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
* **v1.2 (Planned 📋)**: Package renaming (`SwiftSci`), Kalman filter Joseph form fix, byte-level BPE, `addColumn`.
* **v1.3 (Planned 📋)**: Sklearn parity (`ClassificationPipeline`/`RegressionPipeline`, `ColumnTransformer`, `RandomizedSearchCV`, extended metrics, outlier detection).

---

## 📜 License
This project is licensed under the MIT License.
