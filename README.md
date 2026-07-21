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
| **Mean** (1M elements) | 0.071 ms | 0.118 ms | 1.67x | 🟢 Swift |
| **StdDev** (1M elements) | 0.481 ms | 0.499 ms | 1.04x | 🟢 Swift |
| **Variance** (1M elements) | 0.442 ms | 0.499 ms | 1.13x | 🟢 Swift |
| **Pearson Correlation** (500k elements) | 0.781 ms | 1.162 ms | 1.49x | 🟢 Swift |
| **CSV Read** (100k rows, 5 cols) | 166.866 ms | 17.748 ms | 0.11x | 🔴 Python |
| **Filter rows** (100k rows) | 28.006 ms | 0.541 ms | 0.02x | 🔴 Python |
| **GroupBy + sum/mean** (4 groups) | 2.258 ms | 1.538 ms | 0.68x | 🔴 Python |
| **SortBy double column** (100k rows) | 71.059 ms | 6.242 ms | 0.09x | 🔴 Python |
| **LinearRegression fit** (10k×10, 100 epochs) | 27.554 ms | 23.827 ms | 0.86x | 🔴 Python |
| **Random Forest fit** (1k x 4 features, 50 trees) | 4.462 ms | 23.096 ms | 5.18x | 🟢 Swift |
| **GBDT Regressor fit** (1k x 4, 50 estimators) | 33.323 ms | 29.273 ms | 0.88x | 🔴 Python |
| **KMeans fit** (10k x 4, 3 clusters) | 43.779 ms | 11.497 ms | 0.26x | 🔴 Python |
| **PCA SVD fit** (1k×100 → 10 components) | 1.962 ms | 0.744 ms | 0.38x | 🔴 Python |
| **Holt-Winters fit** (50k points, period=12) | 6.503 ms | 134.049 ms | 20.61x | 🟢 Swift |
| **ARIMA fit** (50k points) | 2.304 ms | 201.441 ms | 87.41x | 🟢 Swift |
| **ARIMA forecast horizon=24** (50k points) | 2.273 ms | 202.966 ms | 89.29x | 🟢 Swift |
| **Kalman Filter 1D** (10k observations) | 39.565 ms | 76.823 ms | 1.94x | 🟢 Swift |
| **TS Decomposition additive** (1k points) | 0.448 ms | 0.089 ms | 0.20x | 🔴 Python |
| **LLM Forward Pass** (seqLen=64) | 0.436 ms | 0.499 ms | 1.14x | 🟢 Swift |
| **LLM Generate** (10 tokens) | 6.072 ms | 3.363 ms | 0.55x | 🔴 Python |
| **KernelSHAP Explain** (5 features, 100 coalitions) | 0.153 ms | 0.443 ms | 2.89x | 🟢 Swift |

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

## 📜 License
This project is licensed under the MIT License.
