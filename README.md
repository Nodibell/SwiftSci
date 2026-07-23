# SwiftSci

**SwiftSci** is a native, high-performance, modular data analysis and machine learning library for Swift. It is built from the ground up to leverage Apple Silicon (M-series) unified memory architecture (UMA) and is fully compliant with Swift 6 strict concurrency requirements.

The package combines hardware-accelerated tensor computations on the Apple Silicon GPU via **MLX** with highly optimized CPU vector routines from the **Accelerate framework (vDSP / LAPACK / BLAS)**.

[![Swift Version](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/Nodibell/SwiftSci)
[![Platform Compatibility](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/Nodibell/SwiftSci)
[![Documentation](https://img.shields.io/badge/docs-DocC-blue)](https://nodibell.github.io/SwiftSci/documentation/swiftdataframe/)

---

## 🚀 Core Modules

| Module | Description | Docs |
| :--- | :--- | :---: |
| **`SwiftDataFrame`** | High-performance columnar data manipulation with zero-copy semantics, built on top of `Apache Arrow`. `SystemsCSVParser`, `DataFrame.join`, `pivot`/`melt`, `toFeatureMatrix`/`toTargetVector`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftdataframe/) |
| **`SwiftStats`** | Vectorized descriptive statistics, distributions, and hypothesis tests powered by `Accelerate vDSP`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftstats/) |
| **`SwiftPreprocessing`** | Feature scaling, categorical encoding (`OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`, `FrequencyEncoder`), imputation (`Imputer`, `KNNImputer`), feature selection, `Pipeline`, `ColumnTransformer`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftpreprocessing/) |
| **`SwiftML`** | Linear/Logistic Regression, Decision Trees, Random Forests, GBDTs, `MLPClassifier`/`MLPRegressor`, synthetic generators, Gini importances, `Codable` persistence. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftml/) |
| **`SwiftCluster`** | SVD-based PCA, DBSCAN, `IsolationForest`, `LocalOutlierFactor`, `KMeans` with KMeans++ init. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftcluster/) |
| **`SwiftOptimize`** | `KFold` cross-validation, ROC-AUC, Precision, Recall, F1, `GridSearchCV`, `RandomizedSearchCV`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftoptimize/) |
| **`SwiftForecast`** | Holt-Winters, ARIMA, SARIMA, GARCH, Kalman filter, `ExpandingWindow`, additive/multiplicative decomposition. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftforecast/) |
| **`SwiftNLP`** | `StopWords`, `TextNormalizer`, BPE tokenizer, `NGramTokenizer`, `HashingVectorizer`, `TFIDFVectorizer`, embeddings. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftnlp/) |
| **`SwiftExplain`** | Black-box explainability via parallelized `KernelSHAP`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftexplain/) |
| **`SwiftLLM`** | Local GPU text generation via causal transformer-decoder. SafeTensors & GGUF weight parsing. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftllm/) |
| **`SwiftVisualization`** | Interactive HTML chart exporters: `plotCorrelationHeatmap`, `plotROCCurve`, `plotFeatureImportances`, `plotConfusionMatrix`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftvisualization/) |

---

## 📊 Performance Comparison (Swift vs Python)

The following table presents median execution times on an **Apple Silicon M-series (macOS 15 / arm64)** system, compared directly against popular Python counterparts (Scikit-Learn, NumPy, SHAP, Statsmodels, PyTorch):

| Benchmark Test | Swift (ms) | Python (ms) | Speedup | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Mean** (1M elements) | 0.081 ms | 0.123 ms | 1.52x | 🟢 Swift |
| **StdDev** (1M elements) | 0.446 ms | 0.499 ms | 1.12x | 🟢 Swift |
| **Variance** (1M elements) | 0.466 ms | 0.500 ms | 1.07x | 🟢 Swift |
| **Pearson Correlation** (500k elements) | 0.803 ms | 1.159 ms | 1.44x | 🟢 Swift |
| **CSV Read** (100k rows, 5 cols) | 98.39 ms | 18.05 ms | 0.18x | 🔴 Python |
| **CSV Stream Read** (chunk=10k) | 93.62 ms | 20.36 ms | 0.22x | 🔴 Python |
| **CSV Stream + Filter** | 98.34 ms | 22.59 ms | 0.23x | 🔴 Python |
| **CSV Stream + GroupBy** | 96.02 ms | 25.48 ms | 0.26x | 🔴 Python |
| **Filter rows** (100k rows) | 25.99 ms | 0.53 ms | 0.02x | 🔴 Python |
| **GroupBy + sum/mean** (4 groups) | 2.40 ms | 1.59 ms | 0.66x | 🔴 Python |
| **SortBy double column** (100k rows) | 65.14 ms | 6.50 ms | 0.10x | 🔴 Python |
| **LinearRegression fit** (10k×10, 100 epochs) | 27.12 ms | 23.54 ms | 0.87x | 🔴 Python |
| **Random Forest fit** (1k x 4 features, 50 trees) | 4.84 ms | 24.03 ms | 4.96x | 🟢 Swift |
| **GBDT Regressor fit** (1k x 4, 50 estimators) | 34.40 ms | 30.83 ms | 0.90x | 🔴 Python |
| **KMeans fit** (10k x 4, 3 clusters) | 30.87 ms | 11.87 ms | 0.38x | 🔴 Python |
| **PCA SVD fit** (1k×100 → 10 components) | 2.13 ms | 0.78 ms | 0.37x | 🔴 Python |
| **Holt-Winters fit** (50k points, period=12) | 6.72 ms | 133.76 ms | 19.88x | 🟢 Swift |
| **ARIMA fit** (50k points, Hannan-Rissanen) | 2.36 ms | 201.62 ms | 85.10x | 🟢 Swift |
| **ARIMA forecast horizon=24** (50k points) | 2.46 ms | 202.57 ms | 82.14x | 🟢 Swift |
| **Kalman Filter 1D** (10k observations) | 64.52 ms | 80.45 ms | 1.25x | 🟢 Swift |
| **TS Decomposition additive** (1k points) | 0.358 ms | 0.093 ms | 0.26x | 🔴 Python |
| **LLM Forward Pass** (seqLen=64) | 0.733 ms | 0.457 ms | 0.62x | 🔴 Python |
| **LLM Generate** (10 tokens) | 4.075 ms | 3.540 ms | 0.87x | 🔴 Python |
| **KernelSHAP Explain** (5 features, 100 coalitions) | 0.084 ms | 0.426 ms | 5.07x | 🟢 Swift |

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
* Large tensor operations (Linear/Logistic Regression, Multi-Layer Perceptrons) leverage Apple Silicon GPU execution blocks via `MLXArray` lazy evaluation and Accelerate `BLAS` (`cblas_dgemm`).

### 3. Optional NaN Validation Bypass
We identified that running $O(N)$ CPU sweeps to check for `NaN` presence in statistical calculations adds significant CPU bottlenecks that limit the speed of underlying `vDSP` functions. In v1.0, we introduced a `checkNaN: Bool = true` default parameter to descriptive stats. When bypassed (set to `false` in benchmarks and hot loops), SwiftStats leverages pure hardware SIMD speeds, outperforming NumPy.

---

## 💻 Quick Start

```swift
import Foundation
import SwiftDataFrame
import SwiftPreprocessing
import SwiftML

// 1. Load CSV data with explicit column type overrides
var options = CSVReadOptions()
options.columnTypeOverrides["Survived"] = .int64
let df = try await DataFrame(csv: fileURL, options: options)

// 2. Perform DataFrame operations & extract matrix directly
let cleaned = try df.drop(["PassengerId", "Name", "Ticket", "Cabin"])
let X = try cleaned.toFeatureMatrix(["Pclass", "Age", "Fare"])
let y = try cleaned.toTargetVector("Survived")

// 3. Train Multi-Layer Perceptron (MLP) Neural Network directly on DataFrame
let mlp = MLPClassifier(hiddenLayerSizes: [64, 32], maxIter: 200, learningRate: 0.01, seed: 42)
try await mlp.fit(cleaned, features: ["Age", "Fare"], target: "Survived")
let predictions = try await mlp.predict(cleaned, features: ["Age", "Fare"])
```

---

## 🗺 Roadmap & Future Plans

For detailed implementation plans and ecosystem roadmap, see the [ROADMAP](ROADMAP/ROADMAP.md) directory:
* **v1.1 (Completed 🟢)**: Streaming CSV Parser, SafeTensors & GGUF model loader, SARIMA & GARCH models.
* **v1.2 (Completed 🟢)**: Package renaming (`SwiftSci`), Kalman filter Joseph form fix, byte-level BPE, `addColumn`.
* **v1.3 (Completed 🟢)**: Scikit-Learn Parity (`Pipeline`, `ColumnTransformer`, `RandomizedSearchCV`, `IsolationForest`, `LocalOutlierFactor`, `SMOTE`, `SelectKBest`, `Gini` feature importances).
* **v1.4 (Completed 🟢)**: High-Performance Engine & Quality (`SystemsCSVParser` zero-copy memory-mapped DFA, `RecursiveFeatureElimination` RFE, `Codable` model persistence, `NGramTokenizer`, `HashingVectorizer`, `ExpandingWindow`, `swift-docc-plugin`).
* **v1.5 (Completed 🟢)**: Engine Overhaul & DocC Sprint (Column-parallel CSV reader, `DataFrame.join` hash joins, `pivot`/`melt` matrix reshaping, `MLPClassifier`/`MLPRegressor`, bitmap-free filtering, DocC articles & catalog landing page).
* **v1.6 (Completed 🟢)**: DataFrame ↔ ML Bridge & Hygiene (`toFeatureMatrix`, `toTargetVector`, `columnTypeOverrides`, `StopWords`, `TextNormalizer`, `makeClusters`, `makeCircles`, SPM dependency cleanup).
* **v1.7 (Completed 🟢)**: Performance Sprint, Advanced Encoders & Visualization (`TargetEncoder`, `FrequencyEncoder`, `KNNImputer`, `SwiftVisualization` HTML chart exporters, KMeans 13.8× speedup, CSV Read 1.7× speedup, LLM restored, full DocC site for all 11 modules).

---

## 📜 License
This project is licensed under the MIT License.
