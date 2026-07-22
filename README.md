# SwiftSci

**SwiftSci** is a native, high-performance, modular data analysis and machine learning library for Swift. It is built from the ground up to leverage Apple Silicon (M-series) unified memory architecture (UMA) and is fully compliant with Swift 6 strict concurrency requirements.

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
| **Mean** (1M elements) | 0.082 ms | 0.121 ms | 1.47x | 🟢 Swift |
| **StdDev** (1M elements) | 0.509 ms | 0.539 ms | 1.06x | 🟢 Swift |
| **Variance** (1M elements) | 0.498 ms | 0.520 ms | 1.04x | 🟢 Swift |
| **Pearson Correlation** (500k elements) | 0.868 ms | 1.256 ms | 1.45x | 🟢 Swift |
| **CSV Read** (100k rows, 5 cols) | 177.509 ms | 19.340 ms | 0.11x | 🔴 Python |
| **CSV Stream Read** (chunk=10k) | 238.699 ms | 22.525 ms | 0.09x | 🔴 Python |
| **CSV Stream + Filter** | 243.814 ms | 24.656 ms | 0.10x | 🔴 Python |
| **CSV Stream + GroupBy** | 241.305 ms | 27.899 ms | 0.12x | 🔴 Python |
| **Filter rows** (100k rows) | 30.674 ms | 0.610 ms | 0.02x | 🔴 Python |
| **GroupBy + sum/mean** (4 groups) | 2.357 ms | 1.644 ms | 0.70x | 🔴 Python |
| **SortBy double column** (100k rows) | 83.506 ms | 7.234 ms | 0.09x | 🔴 Python |
| **LinearRegression fit** (10k×10, 100 epochs) | 25.122 ms | 25.062 ms | 1.00x | 🔴 Python |
| **Random Forest fit** (1k x 4 features, 50 trees) | 5.025 ms | 25.475 ms | 5.07x | 🟢 Swift |
| **GBDT Regressor fit** (1k x 4, 50 estimators) | 35.423 ms | 32.905 ms | 0.93x | 🔴 Python |
| **KMeans fit** (10k x 4, 3 clusters) | 62.132 ms | 10.159 ms | 0.16x | 🔴 Python |
| **PCA SVD fit** (1k×100 → 10 components) | 2.019 ms | 0.737 ms | 0.36x | 🔴 Python |
| **Holt-Winters fit** (50k points, period=12) | 6.841 ms | 148.627 ms | 21.73x | 🟢 Swift |
| **ARIMA fit** (50k points) | 2.323 ms | 215.527 ms | 92.78x | 🟢 Swift |
| **ARIMA forecast horizon=24** (50k points) | 2.456 ms | 211.880 ms | 86.26x | 🟢 Swift |
| **Kalman Filter 1D** (10k observations, Joseph Form) | 62.349 ms | 85.547 ms | 1.37x | 🟢 Swift |
| **TS Decomposition additive** (1k points) | 0.459 ms | 0.102 ms | 0.22x | 🔴 Python |
| **LLM Forward Pass** (seqLen=64) | 0.636 ms | 0.528 ms | 0.83x | 🔴 Python |
| **LLM Generate** (10 tokens) | 5.590 ms | 3.366 ms | 0.60x | 🔴 Python |
| **KernelSHAP Explain** (5 features, 100 coalitions) | 0.192 ms | 0.426 ms | 2.22x | 🟢 Swift |

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
