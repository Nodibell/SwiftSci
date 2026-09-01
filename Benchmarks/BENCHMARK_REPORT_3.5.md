# SwiftSci 3.5.0 Benchmark Report

Comprehensive performance verification executed on Apple Silicon (arm64, macOS, Release build) across all 15 core ecosystem modules.

---

## 1. Summary of Benchmark Results

| Benchmark | Target Module | Median (ms) | Min (ms) | Max (ms) | Speedup / Optimization Highlight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Mean (vDSP, 1M elements)`** | `SwiftStats` | **0.082 ms** | 0.079 ms | 0.085 ms | Apple Accelerate single-pass vector reduction |
| **`StdDev (vDSP, 1M elements)`** | `SwiftStats` | **0.309 ms** | 0.298 ms | 0.324 ms | Hardware vectorized standard deviation |
| **`Variance (vDSP, 1M elements)`** | `SwiftStats` | **0.318 ms** | 0.302 ms | 0.341 ms | SIMD variance calculation |
| **`Zero-Allocation df.rows (100k)`** | `SwiftDataFrame` | **12.51 ms** | 12.14 ms | 12.86 ms | Non-allocating struct row view sequence |
| **`toFlatFeatureMatrix (100k rows)`** | `SwiftDataFrame` | **20.30 ms** | 19.99 ms | 20.70 ms | Contiguous 1D flat buffer extraction for ML |
| **`GroupBy + Aggregation`** | `SwiftDataFrame` | **33.15 ms** | 32.85 ms | 33.71 ms | Single-pass hash table index accumulation |
| **`Filter rows (100k)`** | `SwiftDataFrame` | **47.81 ms** | 45.19 ms | 48.53 ms | Vectorized bitmask filtering |
| **`CSV Read (100k rows)`** | `SwiftDataFrame` | **166.75 ms** | 160.95 ms | 169.60 ms | High-throughput columnar CSV parser |
| **`LinearRegression fit (10k x 10)`** | `SwiftML` | **48.92 ms** | 48.70 ms | 50.16 ms | Accelerate BLAS matrix operations |
| **`RandomForest fit (1k x 4, 50 trees)`**| `SwiftML` | **59.36 ms** | 58.32 ms | 61.50 ms | Decision tree bagging with index subsets |
| **`IsolationForest fit (1k x 10)`** | `SwiftCluster` | **89.45 ms** | 88.66 ms | 91.05 ms | Sub-sampled isolation trees |
| **`PCA SVD fit (1k x 100 -> 10 comps)`** | `SwiftCluster` | **110.42 ms** | 110.03 ms | 111.42 ms | LAPACK `dgesvd` singular value decomposition |
| **`Holt-Winters fit (50k pts, p=12)`** | `SwiftForecast` | **75.59 ms** | 75.35 ms | 75.75 ms | Nelder-Mead simplex parameter optimization |
| **`Kalman Filter 1D (10k obs)`** | `SwiftForecast` | **54.25 ms** | 53.79 ms | 54.35 ms | 1D contiguous flat buffer LAPACK `dgesv` |
| **`LLM Forward Pass (seqLen=64)`** | `SwiftLLM` | **1.80 ms** | 1.61 ms | 2.32 ms | MLX Metal Unified Memory RoPE + SwiGLU |
| **`LLM Generate (10 tokens)`** | `SwiftLLM` | **17.55 ms** | 17.13 ms | 21.27 ms | Paged KV-Cache autoregressive token sampling |
| **`KernelSHAP Explain (5 feats, 100)`**| `SwiftExplain` | **0.64 ms** | 0.61 ms | 1.05 ms | Binary coalition mask subset evaluation |
| **`LIME Explain (5 feats, 300)`** | `SwiftExplain` | **2.08 ms** | 1.97 ms | 2.16 ms | Analytical weighted Ridge regression surrogate |
| **`TreeSHAP Explanation (100 samples)`**| `SwiftExplain` | **1.29 ms** | 1.27 ms | 1.35 ms | Polynomial $O(T \cdot L \cdot D^2)$ exact TreeSHAP |
| **`SQLite Direct DataFrame Ingestion`**| `SwiftDatabase` | **0.67 ms** | 0.58 ms | 0.84 ms | Zero-copy columnar SQLite bridge |
| **`UNetSegmentation predict (128x128)`**| `SwiftVision` | **14.32 ms** | 12.12 ms | 15.00 ms | MLX convolution & transposed convolution |

---

## 2. Key Architectural Upgrades in 3.5

1. **Zero-Allocation Row Iteration (`df.rows`)**:
   - Eliminates dictionary heap allocations by referencing column slices through an immutable view struct. Iterates 100,000 rows in ~12 ms.
2. **Flat 1D Buffer Pipeline**:
   - `toFlatFeatureMatrix` maps DataFrame columns into contiguous memory in ~20 ms, feeding BLAS and LAPACK routines with zero reallocation.
3. **Local Surrogate Interpretability (`LIMEExplainer`)**:
   - Generates 300 perturbed instances with Gaussian noise and evaluates weighted Ridge surrogate in ~2 ms.
4. **Hardware Accelerated Statistics (`vDSP`)**:
   - Array summary reductions for 1,000,000 floats executed in under 0.1 ms.
