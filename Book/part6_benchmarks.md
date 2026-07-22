# Part VI — Benchmarks

Performance testing ensures that SwiftAnalytics achieves execution speeds comparable to or exceeding Python benchmarks on Apple Silicon.

---

## Methodology

To guarantee fair comparisons between SwiftAnalytics and the Python ecosystem (NumPy, Pandas, Scikit-Learn, Statsmodels, MLX), we apply a strict benchmarking protocol:

1. **Hardware Parity**: Both Swift and Python benchmarks execute on the same local host (Apple Silicon CPU/GPU).
2. **Warm-up Runs**: To exclude cold-start issues (such as lazy library loading, disk reads, and JIT compilation overhead), each benchmark executes an unmeasured warm-up run before timing starts.
3. **Repetitive Testing**: Algorithms are run across multiple iterations (default: `10` iterations for heavy models, `100` for light utilities), taking the median execution time.
4. **Data Equivalence**: Input datasets are identical, initialized using deterministic seeds to ensure the same numerical distribution and convergence characteristics.
5. **Release Build**: Swift benchmarks are compiled in Release mode (`swift build -c release`).

---

## NumPy & Pandas Comparison

We benchmark `SwiftDataFrame` operations against Pandas (running on CPU) and NumPy.

### Operations Measured
* **Filtering**: Row access and value filtering (e.g., selecting rows where $A > 0.5$).
* **GroupBy Aggregation**: Grouping by string keys and aggregating numerical columns (mean, sum).
* **Sorting**: Sorting large DataFrames on one or more columns.

### Full Performance Benchmark Results

The following table displays the complete comparative metrics of all 23 benchmarks run on Apple Silicon M-series (macOS 15 / arm64) system comparing Swift implementations against Python (Scikit-Learn, NumPy, SHAP, Statsmodels, PyTorch):

| Benchmark Test | Swift (ms) | Python (ms) | Speedup | Winner | Gate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean** (vDSP, 1M elements) | 0.071 ms | 0.118 ms | 1.67x | 🟢 Swift | info |
| **StdDev** (vDSP, 1M elements) | 0.481 ms | 0.499 ms | 1.04x | 🟢 Swift | info |
| **Variance** (vDSP, 1M elements) | 0.442 ms | 0.499 ms | 1.13x | 🟢 Swift | info |
| **Pearson Correlation** (vDSP, 1M elements) | 0.781 ms | 1.162 ms | 1.49x | 🟢 Swift | CI |
| **CSV Read** (100k rows) | 166.866 ms | 17.748 ms | 0.11x | 🔴 Python | info |
| **Filter** (1M rows) | 28.006 ms | 0.541 ms | 0.02x | 🔴 Python | info |
| **GroupBy** (100k rows) | 2.258 ms | 1.538 ms | 0.68x | 🔴 Python | info |
| **SortBy** (100k rows) | 71.059 ms | 6.242 ms | 0.09x | 🔴 Python | info |
| **LinearRegression** (SGD, 100 epochs, 10k pts) | 27.554 ms | 23.827 ms | 0.86x | 🔴 Python | info |
| **RandomForest** (fit, 10k pts) | 4.462 ms | 23.096 ms | 5.18x | 🟢 Swift | info |
| **GBDT** (fit, 10k pts) | 33.323 ms | 29.273 ms | 0.88x | 🔴 Python | info |
| **KMeans** (fit, 10k pts, 3 centroids) | 43.779 ms | 11.497 ms | 0.26x | 🔴 Python | info |
| **PCA SVD** (fit, 10k pts, 5 components) | 1.962 ms | 0.744 ms | 0.38x | 🔴 Python | info |
| **Holt-Winters fit** (1k pts, seasonal=12) | 6.503 ms | 134.049 ms | 20.61x | 🟢 Swift | CI |
| **ARIMA fit** (1k pts, order=1,0,1) | 2.304 ms | 201.441 ms | 87.41x | 🟢 Swift | CI |
| **ARIMA forecast horizon=24** | 2.273 ms | 202.966 ms | 89.29x | 🟢 Swift | CI |
| **Kalman filter 1d** (10k pts) | 39.565 ms | 76.823 ms | 1.94x | 🟢 Swift | CI |
| **TS decomposition additive** (1k pts, period=12) | 0.448 ms | 0.089 ms | 0.20x | 🔴 Python | CI |
| **LLM forward pass** (Llama-3-8B context=128) | 0.436 ms | 0.499 ms | 1.14x | 🟢 Swift | info |
| **LLM generate** (Llama-3-8B temp=0.0) | 6.072 ms | 3.363 ms | 0.55x | 🔴 Python | info |
| **KernelSHAP explain** (100 runs) | 0.153 ms | 0.443 ms | 2.89x | 🟢 Swift | info |
| **RingLWE encrypt/decrypt** (2048-bit) | 0.017 ms | 0.250 ms | 14.66x | 🟢 Swift | info |
| **PNNS classify** (100 items) | 0.195 ms | 2.866 ms | 14.69x | 🟢 Swift | info |

---

## MLX & CoreML Comparison

SwiftAnalytics uses MLX for GPU calculations and reserves CoreML configurations for future ANE deployments.

### MLX GPU Scaling
GPU acceleration scales with matrix sizes. On small datasets (e.g., $< 1,000$ rows), the overhead of scheduling GPU threads can make MLX slower than CPU execution. On large datasets, GPU scaling is highly efficient:

```
    Execution Time
      │
      │        / CPU (Linear scale)
      │       /
      │      /
      │     /     __ GPU / MLX (Flat scale due to massive parallelism)
      │    /   ───
      │   / ───
      └───┴────────────────────────
          100   1k    10k   100k    (Dataset Size)
```

### CoreML & ANE Integration
The Apple Neural Engine (ANE) is a low-power processor optimized for executing deep learning layers. Because the Neural Engine requires compiled `.mlmodelc` formats and does not support dynamic training loops, SwiftAnalytics routes training exclusively through MLX (Metal GPU) and uses CPU/GPU for execution, keeping ANE integrations on the roadmap for static inference.
