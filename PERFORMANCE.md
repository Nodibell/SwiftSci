# SwiftSci 2.4.0 Complete Performance Benchmarks

Official comprehensive comparative benchmark suite results comparing **SwiftSci 2.4.0** against Python data science libraries (**NumPy**, **Pandas**, **Scikit-Learn**, **Statsmodels**, **SHAP**, **PyTorch**) on Apple Silicon (M-series / macOS 15 arm64).

> [!NOTE]
> All benchmarks are executed under identical conditions: deterministic seeds (`seed=42`), single-node execution, and release optimizations (`swift build -c release`).

---

## 📊 Complete Benchmark Matrix

| Benchmark Scenario | SwiftSci 2.4 (Swift) | Python Baseline | Speedup | Winner | Status / Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ARIMA(1,1,1) Fit** (50k pts) | **2.27 ms** | 227.34 ms (*Statsmodels*) | ⚡ **100.1×** | 🟢 **Swift** | Swift 6 native state-space solver |
| **ARIMA(1,1,1) Forecast** (horizon=24) | **2.38 ms** | 224.57 ms (*Statsmodels*) | ⚡ **94.3×** | 🟢 **Swift** | Zero-allocation forecast loop |
| **Holt-Winters Fit** (50k pts, period=12) | **7.35 ms** | 144.90 ms (*Statsmodels*) | ⚡ **19.7×** | 🟢 **Swift** | Vectorized level/trend updates |
| **GroupBy + Aggregation** (100k) | **2.29 ms** | 20.00 ms (*Pandas*) | ⚡ **8.73×** | 🟢 **Swift** | Streaming hash aggregation |
| **RandomForest Fit** (1k×4, 50 trees) | **3.99 ms** | 27.10 ms (*Scikit-Learn*) | ⚡ **6.79×** | 🟢 **Swift** | Pre-sorted DOD trees & SIMD MSE split |
| **GBDT Regressor Fit** (1k×4, 50 est) | **7.99 ms** | 34.80 ms (*Scikit-Learn*) | ⚡ **4.35×** | 🟢 **Swift** | Parallel tree gradient boosting |
| **TreeSHAP Explanation** (100 samples) | **0.33 ms** | 1.20 ms (*SHAP*) | ⚡ **3.63×** | 🟢 **Swift** | Fast decision tree attributions |
| **KernelSHAP Explain** (100 coalitions) | **0.18 ms** | 0.46 ms (*SHAP*) | ⚡ **2.56×** | 🟢 **Swift** | Swift `TaskGroup` parallel coalitions |
| **TF-IDF Vectorizer** (50 docs) | **0.99 ms** | 2.10 ms (*Scikit-Learn*) | ⚡ **2.12×** | 🟢 **Swift** | Parallel term-frequency matrix extraction |
| **LLM Forward Pass** (seqLen=64) | **0.45 ms** | 0.67 ms (*PyTorch*) | ⚡ **1.48×** | 🟢 **Swift** | MLX Metal GPU execution |
| **Kalman Filter 1D** (10k obs) | **60.44 ms** | 87.78 ms (*NumPy*) | ⚡ **1.45×** | 🟢 **Swift** | Accelerate matrix updates |
| **CSV Stream + GroupBy** (100k rows) | **21.21 ms** | 30.05 ms (*Pandas*) | ⚡ **1.41×** | 🟢 **Swift** | Memory-mapped streaming reader |
| **CSV Read** (100k rows) | **14.75 ms** | 20.11 ms (*Pandas*) | ⚡ **1.36×** | 🟢 **Swift** | Memory-mapped zero-copy parser |
| **Mean Reduction** (vDSP 1M elements) | **0.098 ms** | 0.118 ms (*NumPy*) | ⚡ **1.20×** | 🟢 **Swift** | Accelerate `vDSP_meanvD` |
| **StdDev Reduction** (vDSP 1M elements) | **0.433 ms** | 0.516 ms (*NumPy*) | ⚡ **1.19×** | 🟢 **Swift** | Single-pass Accelerate `vDSP` |
| **Pearson Correlation** (500k pairs) | **1.032 ms** | 1.233 ms (*NumPy*) | ⚡ **1.19×** | 🟢 **Swift** | Vectorized dot product |
| **LinearRegression Fit** (10k×10, 100 ep) | **25.88 ms** | 26.10 ms (*Scikit-Learn*) | ⚡ **1.01×** | 🟢 **Swift** | LAPACK OLS analytical solver `dgels_` |
| **SQLite Direct Ingestion** (10k rows) | **0.69 ms** | 0.95 ms (*Pandas*) | ⚡ **1.37×** | 🟢 **Swift** | Direct C-driver `sqlite3_step` binding |
| **CSV Stream Read** (chunk=10k) | **21.38 ms** | 23.35 ms (*Pandas*) | ⚡ **1.09×** | 🟢 **Swift** | Chunked parser |
| **TS Decomposition additive** (1k pts) | **0.252 ms** | 0.10 ms (*Statsmodels*) | 0.40× | 🔴 **Python** | Accelerated via `vDSP_convD` 1D FIR |
| **PCA SVD Fit** (1k×100 → 10 comps) | **1.99 ms** | 0.89 ms (*Scikit-Learn*) | 0.45× | 🔴 **Python** | LAPACK `dgesdd_` vs `ARPACK` randomized |
| **KMeans Fit** (10k×4, 3 clusters) | **22.29 ms** | 12.34 ms (*Scikit-Learn*) | 0.55× | 🔴 **Python** | OpenMP parallel centroids in C |
| **LLM Token Generate** (10 tokens) | **5.80 ms** | 4.70 ms (*PyTorch*) | 0.81× | 🔴 **Python** | Includes streaming UI, Top-K & tokenizer |

---

## 🔍 Detailed Analysis of Optimizations in v2.4.0

1. **SwiftNLP Accelerate & NLTK Equivalence**:
   - VADER Lexicon lookup uses zero-allocation binary search over pre-sorted static Key-Value arrays in `VADERLexicon.swift`.
   - Cosine similarity in `WordEmbeddings` is accelerated via `vDSP_dotprD` and `vDSP_svesqD`.
2. **SIMD Bitmask Filtering (`SwiftDataFrame`)**:
   - `filterIndicesDoubleSIMD` and `filterIndicesInt64SIMD` use `SIMD4` vector registers for fast row mask evaluation.
3. **Primitive Pointer Radix Sorting (`SwiftDataFrame`)**:
   - `sortIndicesPrimitiveFast` for zero-copy array sorting over `UnsafeBufferPointer<Double>`.
4. **vDSP FIR & Spectral Decomposition (`SwiftForecast`)**:
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
