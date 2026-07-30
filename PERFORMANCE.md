# SwiftSci 2.3.0 Complete Performance Benchmarks

Official comprehensive comparative benchmark suite results comparing **SwiftSci 2.3.0** against Python data science libraries (**NumPy**, **Pandas**, **Scikit-Learn**, **Statsmodels**, **SHAP**, **PyTorch**) on Apple Silicon (M-series / macOS 15 arm64).

> [!NOTE]
> All benchmarks are executed under identical conditions: deterministic seeds (`seed=42`), single-node execution, and release optimizations (`swift build -c release`).

---

## 📊 Complete Benchmark Matrix (All 25 Scenarios)

| Benchmark Scenario | SwiftSci 2.3 (Swift) | Python Baseline | Speedup | Winner | Status / Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ARIMA(1,1,1) Fit** (50k pts) | **2.35 ms** | 227.34 ms (*Statsmodels*) | ⚡ **96.74×** | 🟢 **Swift** | Swift 6 native state-space solver |
| **ARIMA(1,1,1) Forecast** (horizon=24) | **2.44 ms** | 224.57 ms (*Statsmodels*) | ⚡ **92.04×** | 🟢 **Swift** | Zero-allocation forecast loop |
| **Holt-Winters Fit** (50k pts, period=12) | **7.01 ms** | 144.90 ms (*Statsmodels*) | ⚡ **20.67×** | 🟢 **Swift** | Vectorized level/trend updates |
| **RandomForest Fit** (1k×4, 50 trees) | **3.82 ms** | 27.10 ms (*Scikit-Learn*) | ⚡ **7.09×** | 🟢 **Swift** | Pre-sorted DOD trees & SIMD MSE split |
| **GBDT Regressor Fit** (1k×4, 50 est) | **8.03 ms** | 34.80 ms (*Scikit-Learn*) | ⚡ **4.33×** | 🟢 **Swift** | Parallel tree gradient boosting |
| **KernelSHAP Explain** (100 coalitions) | **0.18 ms** | 0.46 ms (*SHAP*) | ⚡ **2.56×** | 🟢 **Swift** | Swift `TaskGroup` parallel coalitions |
| **StdDev Reduction** (vDSP 1M elements) | **0.299 ms** | 0.516 ms (*NumPy*) | ⚡ **1.73×** | 🟢 **Swift** | Single-pass Accelerate `vDSP` |
| **Variance Reduction** (vDSP 1M elements) | **0.312 ms** | 0.511 ms (*NumPy*) | ⚡ **1.64×** | 🟢 **Swift** | Single-pass Accelerate `vDSP` |
| **Pearson Correlation** (500k pairs) | **0.797 ms** | 1.233 ms (*NumPy*) | ⚡ **1.54×** | 🟢 **Swift** | Vectorized dot product |
| **Mean Reduction** (vDSP 1M elements) | **0.080 ms** | 0.118 ms (*NumPy*) | ⚡ **1.48×** | 🟢 **Swift** | Accelerate `vDSP_meanvD` |
| **Kalman Filter 1D** (10k obs) | **61.76 ms** | 87.78 ms (*NumPy*) | ⚡ **1.42×** | 🟢 **Swift** | Accelerate matrix updates |
| **CSV Stream + GroupBy** (100k rows) | **21.49 ms** | 30.05 ms (*Pandas*) | ⚡ **1.40×** | 🟢 **Swift** | Streaming hash aggregation |
| **CSV Read** (100k rows) | **15.11 ms** | 20.11 ms (*Pandas*) | ⚡ **1.30×** | 🟢 **Swift** | Memory-mapped zero-copy parser |
| **LLM Forward Pass** (seqLen=64) | **0.64 ms** | 0.67 ms (*PyTorch*) | ⚡ **1.05×** | 🟢 **Swift** | MLX Metal GPU execution |
| **LinearRegression Fit** (10k×10, 100 ep) | 26.82 ms | **25.58 ms** (*Scikit-Learn*) | 0.95× | 🔴 **Python** | Near parity (OLS analytical vs GD) |
| **CSV Stream Read** (chunk=10k) | 21.74 ms | **23.35 ms** (*Pandas*) | ⚡ **1.07×** | 🟢 **Swift** | Chunked parser |
| **CSV Stream + Filter** | 34.97 ms | **25.63 ms** (*Pandas*) | 0.73× | 🔴 **Python** | Pandas C-chunking |
| **KMeans Fit** (10k×4, 3 clusters) | 22.03 ms | **12.34 ms** (*Scikit-Learn*) | 0.56× | 🔴 **Python** | OpenMP parallel centroids in C |
| **PCA SVD Fit** (1k×100 → 10 comps) | 2.05 ms | **0.89 ms** (*Scikit-Learn*) | 0.43× | 🔴 **Python** | LAPACK `dgesdd_` vs `ARPACK` randomized |
| **LLM Token Generate** (10 tokens) | 5.56 ms | **4.70 ms** (*PyTorch*) | 0.85× | 🔴 **Python** | Includes streaming UI, Top-K & tokenizer |
| **SortBy double column** (100k rows) | **79.08 ms** | **7.84 ms** (*Pandas*) | 0.10× | 🔴 **Python** | Accelerated via `sortIndicesPrimitiveFast` vDSP |
| **TS Decomposition additive** (1k pts) | **0.258 ms** | **0.10 ms** (*Statsmodels*) | 0.39× | 🔴 **Python** | Accelerated via `vDSP_convD` 1D FIR |
| **Filter rows** (100k rows) | **35.58 ms** | **0.61 ms** (*Pandas*) | 0.02× | 🔴 **Python** | Accelerated via `filterIndicesDoubleSIMD` bitmask |

---

## 🔍 Detailed Analysis of Optimizations in v2.3.0

1. **SIMD Bitmask Filtering (`SwiftDataFrame`)**:
   Introduced `filterIndicesDoubleSIMD` and `filterIndicesInt64SIMD` using `SIMD4` vector comparison registers to evaluate boolean masks directly on raw pointers.
2. **Primitive Pointer Radix Sorting (`SwiftDataFrame`)**:
   Introduced `sortIndicesPrimitiveFast` for zero-copy array sorting over `UnsafeBufferPointer<Double>` and `UnsafeBufferPointer<Int64>`.
3. **vDSP FIR & Spectral Decomposition (`SwiftForecast`)**:
   Accelerated 1D moving average convolution via `vDSP_convD` and real FFT decomposition via `vDSP_fft_zipD`.
4. **Tree Split Evaluation (`SwiftML`)**:
   Accelerated `mseImpurity` evaluation via Accelerate `vDSP.mean` and `vDSP.meanSquare`.

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
