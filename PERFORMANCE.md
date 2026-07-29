# SwiftSci 2.2.0 Complete Performance Benchmarks

Official comprehensive comparative benchmark suite results comparing **SwiftSci 2.2.0** against Python data science libraries (**NumPy**, **Pandas**, **Scikit-Learn**, **Statsmodels**, **SHAP**, **PyTorch**) on Apple Silicon (M-series / macOS 15 arm64).

> [!NOTE]
> All benchmarks are executed under identical conditions: deterministic seeds (`seed=42`), single-node execution, and release optimizations (`swift build -c release`).

---

## 📊 Complete Benchmark Matrix (All 25 Scenarios)

| Benchmark Scenario | SwiftSci 2.2 (Swift) | Python Baseline | Speedup | Winner | Status / Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ARIMA(1,1,1) Fit** (50k pts) | **2.48 ms** | 227.34 ms (*Statsmodels*) | ⚡ **91.60×** | 🟢 **Swift** | Swift 6 native state-space solver |
| **ARIMA(1,1,1) Forecast** (horizon=24) | **2.49 ms** | 224.57 ms (*Statsmodels*) | ⚡ **90.33×** | 🟢 **Swift** | Zero-allocation forecast loop |
| **Holt-Winters Fit** (50k pts, period=12) | **7.42 ms** | 144.90 ms (*Statsmodels*) | ⚡ **19.52×** | 🟢 **Swift** | Vectorized level/trend updates |
| **RandomForest Fit** (1k×4, 50 trees) | **4.63 ms** | 27.10 ms (*Scikit-Learn*) | ⚡ **5.86×** | 🟢 **Swift** | Pre-sorted feature matrix DOD trees |
| **GBDT Regressor Fit** (1k×4, 50 est) | **8.51 ms** | 34.80 ms (*Scikit-Learn*) | ⚡ **4.09×** | 🟢 **Swift** | Parallel tree gradient boosting |
| **KernelSHAP Explain** (100 coalitions) | **0.18 ms** | 0.46 ms (*SHAP*) | ⚡ **2.57×** | 🟢 **Swift** | Swift `TaskGroup` parallel coalitions |
| **StdDev Reduction** (vDSP 1M elements) | **0.311 ms** | 0.516 ms (*NumPy*) | ⚡ **1.66×** | 🟢 **Swift** | Single-pass `vDSP_measqvD` |
| **Variance Reduction** (vDSP 1M elements) | **0.318 ms** | 0.511 ms (*NumPy*) | ⚡ **1.61×** | 🟢 **Swift** | Single-pass `vDSP_measqvD` |
| **Mean Reduction** (vDSP 1M elements) | **0.082 ms** | 0.118 ms (*NumPy*) | ⚡ **1.44×** | 🟢 **Swift** | Accelerate `vDSP_meanvD` |
| **Pearson Correlation** (500k pairs) | **0.866 ms** | 1.233 ms (*NumPy*) | ⚡ **1.42×** | 🟢 **Swift** | Vectorized dot product |
| **LLM Forward Pass** (seqLen=64) | **0.51 ms** | 0.67 ms (*PyTorch*) | ⚡ **1.31×** | 🟢 **Swift** | MLX Metal GPU execution |
| **CSV Stream + GroupBy** (100k rows) | **22.88 ms** | 30.05 ms (*Pandas*) | ⚡ **1.31×** | 🟢 **Swift** | Streaming hash aggregation |
| **CSV Read** (100k rows) | **16.53 ms** | 20.11 ms (*Pandas*) | ⚡ **1.22×** | 🟢 **Swift** | Memory-mapped zero-copy parser |
| **Kalman Filter 1D** (10k obs) | 96.00 ms | **87.78 ms** (*NumPy*) | 0.91× | 🔴 **Python** | Near parity |
| **CSV Stream Read** (chunk=10k) | 26.25 ms | **23.35 ms** (*Pandas*) | 0.89× | 🔴 **Python** | Near parity |
| **CSV Stream + Filter** | 35.59 ms | **25.63 ms** (*Pandas*) | 0.72× | 🔴 **Python** | Pandas C-chunking |
| **LinearRegression Fit** (10k×10, 100 ep) | 35.63 ms | **25.58 ms** (*Scikit-Learn*) | 0.72× | 🔴 **Python** | GD iteration overhead (OLS O(1) is fast) |
| **KMeans Fit** (10k×4, 3 clusters) | 23.64 ms | **12.34 ms** (*Scikit-Learn*) | 0.52× | 🔴 **Python** | OpenMP parallel centroids in C |
| **PCA SVD Fit** (1k×100 → 10 comps) | 2.20 ms | **0.89 ms** (*Scikit-Learn*) | 0.40× | 🔴 **Python** | LAPACK `dgesdd_` vs `ARPACK` randomized |
| **LLM Token Generate** (10 tokens) | 12.73 ms | **4.70 ms** (*PyTorch*) | 0.37× | 🔴 **Python** | Swift includes streaming UI, Top-K & tokenizer |
| **SortBy double column** (100k rows) | 80.34 ms | **7.84 ms** (*Pandas*) | 0.10× | 🔴 **Python** | Tracked for v2.3 radix sort optimization |
| **TS Decomposition additive** (1k pts) | 1.15 ms | **0.10 ms** (*Statsmodels*) | 0.09× | 🔴 **Python** | FFT convolution vs loop |
| **Filter rows** (100k rows) | 30.03 ms | **0.61 ms** (*Pandas*) | 0.02× | 🔴 **Python** | Tracked for v2.3 SIMD index mask optimization |

---

## 🔍 Detailed Analysis of Performance Gaps & Future Optimizations (v2.3)

1. **DataFrame Row Filtering (`0.02×`)**:
   Pandas uses contiguous C bitmasks and NumPy boolean indexing. In SwiftSci, v2.2 introduced reusable `DataFrameRow` instances eliminating 100k heap allocations; v2.3 will introduce SIMD-vectorized bitmask index gathers to achieve parity.
2. **DataFrame Sorting (`0.10×`)**:
   Pandas delegates to C-level `argsort` radix sort on primitive arrays. SwiftSci currently sorts via `parallelGathered(at:)`; v2.3 will add Accelerate `vDSP.sort` radix indexing.
3. **LLM Generation Overhead (`0.37×`)**:
   PyTorch benchmark measures raw tensor `argmax()`. SwiftSci measures the complete production RAG/LLM pipeline (Task creation, Temperature + Top-K 50 sampling, BPE string decoding, and `AsyncStream` token yielding).

---

## 🖥️ Benchmark Platform Details

- **Hardware**: Apple Silicon M-series (Unified Memory Architecture - UMA)
- **Swift**: Swift 6 (Strict Concurrency Enabled, Accelerated via `vDSP` / `LAPACK` & `MLX`)
- **Python**: 3.11.9 (`NumPy 2.3.5`, `Pandas 3.0.2`, `Scikit-Learn 1.4`, `Statsmodels 0.14`, `PyTorch 2.11`, `SHAP 0.44`)

---

## 🛠️ How to Reproduce

Run native Swift release benchmarks:
```bash
swift run -c release SwiftAnalyticsBenchmarks --json Benchmarks/Results/swift_results.json
```

Run Python comparison suite:
```bash
python3 Benchmarks/Python/benchmarks.py --json Benchmarks/Results/python_results.json
python3 Benchmarks/Python/compare.py Benchmarks/Results/swift_results.json Benchmarks/Results/python_results.json
```
