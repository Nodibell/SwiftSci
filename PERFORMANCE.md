# SwiftSci 3.0.0 Complete Performance Benchmarks

Official comprehensive comparative benchmark suite results comparing **SwiftSci 3.0.0** (Release Build `-c release`) against Python data science libraries (**NumPy**, **Pandas**, **Scikit-Learn**, **Statsmodels**, **SHAP**, **PyTorch**, **Ultralytics**) on Apple Silicon (M-series / macOS 15 arm64).


> [!NOTE]
> All benchmarks are executed under identical conditions: deterministic seeds (`seed=42`), single-node execution, and release optimizations (`swift run -c release`).

---

## 📊 Complete Benchmark Matrix

The values below are median times from the latest release benchmark run. Speedups are computed as `Python / Swift`; values above `1×` favor Swift. `n/a` means that the Python suite did not include an equivalent benchmark.

| Benchmark Scenario | SwiftSci 3.0.0 (Swift) | Python Baseline | Speedup | Winner | Status / Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Mean (1M elements) | **0.082 ms** | 0.121 ms (*NumPy*) | ⚡ **1.48×** | 🟢 **Swift** | vDSP reduction |
| StdDev (1M elements) | **0.275 ms** | 0.533 ms (*NumPy*) | ⚡ **1.94×** | 🟢 **Swift** | vDSP reduction |
| Variance (1M elements) | **0.282 ms** | 0.517 ms (*NumPy*) | ⚡ **1.84×** | 🟢 **Swift** | vDSP reduction |
| Pearson Correlation (500k pairs) | **0.812 ms** | 1.193 ms (*NumPy*) | ⚡ **1.47×** | 🟢 **Swift** | CI-gated |
| CSV Read (100k rows) | **15.465 ms** | 19.413 ms (*Pandas*) | ⚡ **1.26×** | 🟢 **Swift** | Memory-mapped parser |
| CSV Stream Read (chunk=10k) | **21.715 ms** | 21.921 ms (*Pandas*) | ⚡ **1.01×** | 🟢 **Swift** | Near parity |
| CSV Stream + Filter | **29.069 ms** | 24.233 ms (*Pandas*) | 0.83× | 🔴 **Python** | Informational gap |
| CSV Stream + GroupBy | **22.830 ms** | 27.603 ms (*Pandas*) | ⚡ **1.21×** | 🟢 **Swift** | Streaming group-by |
| Filter rows (100k) | **32.378 ms** | 0.581 ms (*Pandas*) | 0.02× | 🔴 **Python** | Informational gap |
| GroupBy + Aggregation | **2.411 ms** | n/a | n/a | — | Swift-only benchmark |
| SortBy double column | **68.424 ms** | 7.210 ms (*Pandas*) | 0.11× | 🔴 **Python** | Informational gap |
| LinearRegression fit (10k×10, 100 epochs) | **26.063 ms** | 24.921 ms (*Scikit-Learn*) | 0.96× | 🔴 **Python** | Near parity; informational |
| RandomForest fit (1k×4, 50 trees) | **3.838 ms** | 25.300 ms (*Scikit-Learn*) | ⚡ **6.59×** | 🟢 **Swift** | CI-gated |
| GBDT Regressor fit (1k×4, 50 estimators) | **8.397 ms** | 32.366 ms (*Scikit-Learn*) | ⚡ **3.85×** | 🟢 **Swift** | CI-gated |
| KMeans fit (10k×4, 3 clusters) | **20.062 ms** | 11.993 ms (*Scikit-Learn*) | 0.60× | 🔴 **Python** | Informational gap |
| PCA SVD fit (1k×100 → 10 comps) | **1.014 ms** | 0.732 ms (*Scikit-Learn*) | 0.72× | 🔴 **Python** | Informational gap |
| IsolationForest fit (1k×10, 100 trees) | **14.794 ms** | n/a | n/a | — | Swift-only benchmark |
| LinearSVC fit (1k×4, 100 epochs, Metal GPU) | **0.463 ms** | n/a | n/a | — | Swift MLX Metal benchmark |
| Holt-Winters fit (50k pts, period=12) | **6.451 ms** | 144.752 ms (*Statsmodels*) | ⚡ **22.44×** | 🟢 **Swift** | CI-gated |
| ARIMA(1,1,1) fit (50k pts) | **2.463 ms** | 212.621 ms (*Statsmodels*) | ⚡ **86.34×** | 🟢 **Swift** | CI-gated |
| ARIMA(1,1,1) forecast (horizon=24) | **2.566 ms** | 213.709 ms (*Statsmodels*) | ⚡ **83.27×** | 🟢 **Swift** | CI-gated |
| Kalman Filter 1D (10k observations) | **57.970 ms** | 85.788 ms (*NumPy*) | ⚡ **1.48×** | 🟢 **Swift** | CI-gated |
| TS Decomposition additive (1k pts) | **0.255 ms** | 0.100 ms (*Statsmodels*) | 0.39× | 🔴 **Python** | CI-gated, still below 2× threshold |
| LLM Forward Pass (seqLen=64) | **0.437 ms** | 0.531 ms (*PyTorch*) | ⚡ **1.21×** | 🟢 **Swift** | MLX Metal execution |
| LLM Generate (10 tokens) | **4.936 ms** | 3.609 ms (*PyTorch*) | 0.73× | 🔴 **Python** | Informational gap |
| YOLOv8Detector detect (640×640 real GPU) | **12.317 ms** | n/a | n/a | — | SwiftVision GPU benchmark |
| YOLOPreprocessor letterbox (1920×1080 → 640×640) | **0.414 ms** | n/a | n/a | — | SwiftVision preprocessing |
| UNetSegmentationModel predict (128×128 image) | **0.107 ms** | n/a | n/a | — | SwiftVision GPU benchmark |
| KernelSHAP Explain (5 features, 100 coalitions) | **0.168 ms** | 0.413 ms (*SHAP*) | ⚡ **2.45×** | 🟢 **Swift** | CI-gated |
| CNN Feature Extraction & Vision Metrics | **0.004 ms** | n/a | n/a | — | SwiftSci Extensions benchmark |
| SQLite Direct DataFrame Ingestion | **0.776 ms** | n/a | n/a | — | SwiftSci Extensions benchmark |
| RAG Context Summary Generation | **0.000 ms** | n/a | n/a | — | SwiftSci Extensions benchmark |
| TreeSHAP Explanation (100 samples) | **0.310 ms** | n/a | n/a | — | SwiftSci Extensions benchmark |
| OneVsRestClassifier (5 classes, 100 samples) | **3.566 ms** | n/a | n/a | — | SwiftSci Extensions benchmark |
| TF-IDF Vectorizer (50 documents) | **0.590 ms** | n/a | n/a | — | SwiftSci Extensions benchmark |

**Comparison summary:** 15 matched benchmarks favor Swift; 8 favor Python. The CI comparison passed with no gated regressions.



---

## 🔍 Detailed Analysis of Optimizations in v3.0.0

1. **`SwiftPreprocessing` Zero-Allocation Value Semantics**:
   - `MinMaxScaler`, `StandardScaler`, and `RobustScaler` operate as thread-safe `struct` value types.
   - `Pipeline` and `ColumnTransformer` mutate array elements directly by index in-place.
2. **SwiftNLP WordNet Engine & Accelerate Optimization**:
   - WordNet synset lookup and similarity metrics (`pathSimilarity`, `wupSimilarity`) leverage fast in-memory trie structures.
   - VADER Lexicon lookup uses zero-allocation binary search over pre-sorted static Key-Value arrays in `VADERLexicon.swift`.
   - Cosine similarity in `WordEmbeddings` is accelerated via `vDSP_dotprD` and `vDSP_svesqD`.
3. **SIMD Bitmask Filtering (`SwiftDataFrame`)**:
   - `filterIndicesDoubleSIMD` and `filterIndicesInt64SIMD` use `SIMD4` vector registers for fast row mask evaluation.
4. **Primitive Pointer Radix Sorting (`SwiftDataFrame`)**:
   - `sortIndicesPrimitiveFast` for zero-copy array sorting over `UnsafeBufferPointer<Double>`.
5. **vDSP FIR & Spectral Decomposition (`SwiftForecast`)**:
   - 1D moving average convolution via `vDSP_convD` and real FFT decomposition via `vDSP_fft_zipD`.

---

## 🖥️ Benchmark Platform Details

- **Hardware**: Apple Silicon M-series (Unified Memory Architecture - UMA)
- **Swift**: Swift 6 (Strict Concurrency Enabled, Accelerated via `vDSP` / `LAPACK` & `MLX`)
- **Python**: 3.11.9 (`NumPy 2.3.5`, `Pandas 3.0.2`, `Scikit-Learn 1.4`, `Statsmodels 0.14`, `PyTorch 2.11`, `SHAP 0.44`)

---

## 🛠️ How to Reproduce

Run native Swift release benchmarks:
```bash
swift run -c release SwiftSciBenchmarks --json Benchmarks/Results/swift_results.json
```

Run Python comparison suite:
```bash
python3 Benchmarks/Python/benchmarks.py --json Benchmarks/Results/python_results.json
python3 Benchmarks/Python/compare.py Benchmarks/Results/swift_results.json Benchmarks/Results/python_results.json
```
