# Implementation Plan 17: Performance Optimization Sprint, Advanced Feature Engineering & Interactive Visualizations (v1.7)

**Target Version:** `v1.7.0`  
**Focus:** Performance Regression Fixes (KMeans, CSV Read, LLM), Advanced Categorical Encoders, Imputation & Visualizations  
**Branch:** `feature/v1.7.0-performance-encoders-viz`  

---

## 1. Executive Summary & Benchmark Regression Analysis (v1.4 → v1.6.1)

Empirical benchmarking of **v1.6.1** against **v1.4.0** revealed significant performance highlights alongside critical regressions that must be addressed prior to new feature delivery.

### 📊 Benchmark Comparison Summary

| Benchmark | v1.4 (ms) | v1.6.1 (ms) | Change | Status / Root Cause |
| :--- | :---: | :---: | :---: | :--- |
| **KMeans fit (10k×4)** | `44.7` | `413.4` | 🔴 **9.2× Slower** | `HardwareRouter` misrouted 10k×4 matrix to GPU (`MLX`), causing massive `flatMap` + `Float` conversion & GPU dispatch latency. |
| **CSV Read (100k rows)** | `71.5` | `144.9` | 🔴 **2.0× Slower** | Bulk parser pathway was unified into row-by-row streaming iterator, introducing extra allocation per row. |
| **LLM Forward Pass** | `0.429` | `0.809` | 🔴 **1.9× Slower** | Eager MLXArray allocations and non-cached kv-cache evaluation. |
| **LLM Generate (10 tokens)** | `4.07` | `5.11` | 🔴 **1.25× Slower** | Token generation loop allocations. |
| **CSV Stream Read** | `276.8` | `161.5` | 🟢 **1.7× Faster** | Optimized chunked buffer parsing. |
| **CSV Stream + GroupBy** | `277.9` | `148.3` | 🟢 **1.9× Faster** | Aggregation loop optimizations. |
| **SortBy** | `73.5` | `64.0` | 🟢 **13% Faster** | Accelerate vDSP sort integration. |
| **KernelSHAP** | `0.191` | `0.172` | 🟢 **2.77× Faster vs Python** | Parallel coalition evaluation. |
| **ARIMA (50k pts)** | `2.24` | `2.45` | 🟢 **82–90× Faster vs Python** | OLS Hannan-Rissanen LAPACK engine. |
| **Holt-Winters (50k pts)** | `6.40` | `6.65` | 🟢 **21× Faster vs Python** | Native SIMD vectorization. |

---

## 2. Priority 0: Critical Performance Optimization Sprint

### A. KMeans 9.2× Regression Fix (`Sources/SwiftCluster/Core/KMeans.swift`)
1. **Hardware Routing Threshold Adjustment**:
   - Re-evaluate `HardwareRouter` routing threshold for `KMeans`. Small/medium matrices ($N \le 50k$) will default to high-performance CPU vDSP / flat array memory layout.
2. **Eliminate Allocation Overhead**:
   - Avoid `features.flatMap { $0.map { Float($0) } }` inside loop.
   - Use `withUnsafeBufferPointer` and contiguous flat `[Double]` storage for centroids and samples.

### B. Bulk CSV Reading 2× Optimization (`Sources/SwiftDataFrame/IO/CSVReader.swift`)
1. **Dedicated Fast-Path for `CSVReader.read`**:
   - Separate bulk memory-mapped parsing from the row-by-row streaming iterator.
   - Use vectorized string scanners and direct column allocation to restore CSV reading time to $< 70\text{ ms}$.

### C. LLM Forward & Generation Optimization (`Sources/SwiftLLM`)
1. **MLX Array Reuse & KV-Cache Warmup**:
   - Implement persistent allocation buffers for KV-Cache in `TransformerBlock`.
   - Prevent redundant tensor reshapes during sequence generation.

---

## 3. Priority 1: Advanced Preprocessing & Feature Engineering (`SwiftPreprocessing`)

### A. Categorical Encoders
1. **`TargetEncoder`** (`Sources/SwiftPreprocessing/Encoders/TargetEncoder.swift`):
   - Computes smoothed target mean for high-cardinality categories:
     $$S(c) = \frac{n_c \cdot \bar{y}_c + m \cdot \bar{y}_{global}}{n_c + m}$$
2. **`FrequencyEncoder`** (`Sources/SwiftPreprocessing/Encoders/FrequencyEncoder.swift`):
   - Encodes categories by normalized occurrence frequency ($n_c / N$).

### B. Advanced Imputation
1. **`KNNImputer`** (`Sources/SwiftPreprocessing/Imputation/KNNImputer.swift`):
   - Distance-weighted $k$-Nearest Neighbors missing value imputation for numerical matrices.

---

## 4. Priority 2: Interactive Visualization Exporters (`SwiftVisualization`)

* **Module**: `SwiftVisualization`
* **Exporters**:
  - `plotCorrelationHeatmap(df)`
  - `plotROCCurve(yTrue, yScore)`
  - `plotFeatureImportances(names, importances)`
  - `plotConfusionMatrix(yTrue, yPred)`
* Generates standalone HTML files with embedded Plotly.js charts for interactive browser exploration.

---

## 5. Verification & Benchmark Gating Plan

1. **Benchmark Suite Execution**:
   - Run `python3 Benchmarks/Python/compare.py` to confirm:
     - KMeans fit $< 40\text{ ms}$ (Target: $\ge 1.0\times$ vs Python).
     - CSV Read $< 70\text{ ms}$.
     - LLM Forward Pass $< 0.45\text{ ms}$.
2. **Unit Tests**:
   - Run `swift test` across all targets to maintain 100% pass rate.
3. **DocC Documentation**:
   - Verify DocC catalog compilation via `swift package generate-documentation`.
