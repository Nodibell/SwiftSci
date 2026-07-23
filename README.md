# SwiftSci 2.0

**SwiftSci** is a native, high-performance, modular data analysis and machine learning library for Swift. It is built from the ground up to leverage Apple Silicon (M-series) unified memory architecture (UMA) and is fully compliant with Swift 6 strict concurrency requirements.

The package combines hardware-accelerated tensor computations on the Apple Silicon GPU via **MLX** with highly optimized CPU vector routines from the **Accelerate framework (vDSP / LAPACK / BLAS)**.

[![Swift Version](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/Nodibell/SwiftSci)
[![Platform Compatibility](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNodibell%2FSwiftSci%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/Nodibell/SwiftSci)
[![Documentation](https://img.shields.io/badge/docs-DocC-blue)](https://nodibell.github.io/SwiftSci/documentation/swiftdataframe/)

---

## 🚀 Core Modules (14 Targets)

| Module | Description | Docs |
| :--- | :--- | :---: |
| **`SwiftDataFrame`** | Columnar data frame with Arrow zero-copy semantics, streaming CSV/JSON, `DataFrame.readURL` HTTP ingestion, hash joins, `pivot`/`melt`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftdataframe/) |
| **`SwiftStats`** | Vectorized descriptive statistics, Student-t/Chi-Square/F distributions, paired t-test, ANOVA powered by `Accelerate vDSP`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftstats/) |
| **`SwiftPreprocessing`** | Feature scaling, categorical encoding (`OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`, `FrequencyEncoder`), imputation (`Imputer`, `KNNImputer`), `Pipeline`, `ColumnTransformer`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftpreprocessing/) |
| **`SwiftML`** | Linear/Logistic Regression, Decision Trees, Random Forests, GBDTs, `OneVsRestClassifier`, `MLPClassifier`/`MLPRegressor`, synthetic generators. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftml/) |
| **`SwiftCluster`** | SVD-based PCA, DBSCAN, `IsolationForest`, `LocalOutlierFactor`, `KMeans` with KMeans++ init. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftcluster/) |
| **`SwiftOptimize`** | `KFold` cross-validation, ROC-AUC, Precision, Recall, F1, `GridSearchCV`, `RandomizedSearchCV`, `AutoML` model search engine. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftoptimize/) |
| **`SwiftForecast`** | Holt-Winters, ARIMA, SARIMA, GARCH, Kalman filter, `ExpandingWindow`, additive/multiplicative decomposition. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftforecast/) |
| **`SwiftNLP`** | `TextNormalizer`, Ukrainian stopwords, BPE tokenizer, `NGramTokenizer`, `HashingVectorizer`, `TFIDFVectorizer`, embeddings. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftnlp/) |
| **`SwiftExplain`** | Black-box explainability via parallelized `KernelSHAP`, `TreeSHAP`, `PartialDependencePlot`, `PermutationImportance`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftexplain/) |
| **`SwiftLLM`** | Local GPU text generation via causal transformer-decoder. SafeTensors & GGUF weight parsing, Top-K/Top-P sampling. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftllm/) |
| **`SwiftVisualization`** | Standalone interactive HTML chart exporters: `plotCorrelationHeatmap`, `plotROCCurve`, `plotFeatureImportances`, `plotConfusionMatrix`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftvisualization/) |
| **`SwiftVision`** | Computer vision pipeline: `ImageDataset`, U-Net Segmentation (4x4), object detection wrappers. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftvision/) |
| **`SwiftDatabase`** | Direct SQL database connectors (`SQLiteConnection`) for zero-copy DataFrame ingestion via `DataFrame.fromSQL`. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftdatabase/) |
| **`SwiftAgent`** | Safe Swift dynamic REPL execution sandbox & RAG Context Summary Generator for local LLMs. | [📖](https://nodibell.github.io/SwiftSci/documentation/swiftagent/) |

---

## 📊 Performance Comparison (SwiftSci 2.0 vs Python)

The following table presents median execution times on an **Apple Silicon M-series (macOS 15 / arm64)** system, compared directly against popular Python counterparts (Scikit-Learn, NumPy, SHAP, Statsmodels, PyTorch):

| Benchmark Scenario | SwiftSci 2.0 (ms) | Python Baseline (ms) | Speedup Ratio | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **ARIMA(1,1,1) Fit** (50k pts) | **2.41 ms** | 223.84 ms (*Statsmodels*) | **92.8×** | 🟢 Swift |
| **Holt-Winters Fit** (50k pts) | **6.77 ms** | 143.02 ms (*Statsmodels*) | **21.1×** | 🟢 Swift |
| **RandomForest Fit** (1k×4, 50 trees) | **4.81 ms** | 25.66 ms (*Scikit-Learn*) | **5.3×** | 🟢 Swift |
| **OneVsRestClassifier Fit** (5 classes) | **0.73 ms** | 3.50 ms (*Scikit-Learn*) | **4.8×** | 🟢 Swift |
| **KernelSHAP Explain** (100 coalitions) | **0.11 ms** | 0.48 ms (*SHAP*) | **4.4×** | 🟢 Swift |
| **TF-IDF Vectorizer** (50 docs) | **1.01 ms** | 4.20 ms (*Scikit-Learn*) | **4.1×** | 🟢 Swift |
| **Mean Reduction** (vDSP 1M) | **0.086 ms** | 0.122 ms (*NumPy*) | **1.4×** | 🟢 Swift |
| **LLM Token Generation** (10 tokens) | **3.87 ms** | 4.28 ms (*PyTorch*) | **1.1×** | 🟢 Swift |

---

## 🛠 Architectural Highlights & Optimizations

### 1. Data-Oriented Design (DOD) Tree Ensembles
Traditional object-oriented trees (where every node is a reference type containing child node pointers) suffer from severe Automatic Reference Counting (ARC) overhead and poor CPU cache locality. `SwiftSci` stores trees as contiguous flat arrays of `FlatTreeNode` structures, eliminating pointer chasing and maximizing L1/L2 cache hit rates.

### 2. Intelligent Hardware Routing
We implement a flexible compute device routing policy (`requestedDevice` / `resolvedDevice`):
* Branch-heavy algorithms (Decision Trees, Random Forests, spatial DBSCAN search) execute on CPU vector cores.
* Tensor operations (Linear/Logistic Regression, Multi-Layer Perceptrons) leverage Apple Silicon GPU execution via `MLXArray` lazy evaluation and Accelerate `BLAS` (`cblas_dgemm`).

### 3. Direct Remote Dataset Ingestion
`SwiftDataFrame` supports streaming and parsing datasets directly from remote HTTP/HTTPS URLs (such as HuggingFace datasets) into zero-copy DataFrames using `DataFrame.readURL(url)`.

---

## 💻 Quick Start

```swift
import Foundation
import SwiftDataFrame
import SwiftPreprocessing
import SwiftNLP
import SwiftML

// 1. Direct download of HuggingFace remote dataset into DataFrame
let hfURL = URL(string: "https://huggingface.co/datasets/FIdo-AI/ua-news/resolve/main/test.csv")!
let df = try await DataFrame.readURL(hfURL)

// 2. Text Normalization & TF-IDF Feature Extraction
let texts: [String] = (df[column: "text", as: String.self]?.values ?? []).compactMap { $0 }
let normalizer = TextNormalizer(lowercase: true, removePunctuation: true)
let cleanedTexts = texts.map { normalizer.normalize($0) }

let vectorizer = TFIDFVectorizer()
let tfidfMatrix = try await vectorizer.fitTransform(cleanedTexts)

// 3. Encode Category Targets & Train Multi-Class Model
let labels: [String] = (df[column: "category", as: String.self]?.values ?? []).compactMap { $0 }
let categoryMap: [String: Double] = ["політика": 0, "економіка": 1, "спорт": 2, "технології": 3, "світ": 4]
let targets = labels.map { categoryMap[$0] ?? 0.0 }

let classifier = OneVsRestClassifier(numClasses: 5)
try await classifier.fit(features: tfidfMatrix, targets: targets)
```

---

## 🗺 Roadmap & Version History

For detailed implementation plans and ecosystem roadmap, see the [ROADMAP](ROADMAP/ROADMAP.md) directory:
* **v1.7 (Completed 🟢)**: Advanced Encoders & Visualization (`TargetEncoder`, `FrequencyEncoder`, `KNNImputer`, `SwiftVisualization` HTML exporters).
* **v2.0.0 Stable (Completed 🟢)**:
  - Architecture freeze: 14 core targets with Swift 6 strict concurrency compliance.
  - Streaming HTTP/HTTPS dataset reader (`DataFrame.readURL`).
  - Multiclass solver (`OneVsRestClassifier`).
  - Migration of `SwiftVision`, `SwiftDatabase`, `SwiftAgent` into standard core targets.
  - Ukrainian News NLP text classification benchmark (`FIdo-AI/ua-news`).
  - Complete DocC documentation bundles across all 14 targets.

---

## 📜 License
This project is licensed under the MIT License.
