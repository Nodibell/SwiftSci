# SwiftSci 3.10.3 historical performance benchmarks

Use the [standardized benchmark guide](Benchmarks/README.md) for current runs, audits and comparisons. The results below are historical research records. They are unvalidated under the standardized contracts and make no certification or production claims. Learned-model and GPU entries remain research unless a separately audited workload covers them. Preserve their original values when reproducing or discussing this report.

Historical research measurements for SwiftSci 3.10.3 Release builds alongside Python data-science libraries including NumPy, Pandas, SciPy, Scikit-Learn, Statsmodels, SHAP, PyTorch, and MLX on Apple Silicon / macOS 15 arm64.

> [!NOTE]
> **Benchmark Methodology v2**
>
> SwiftSci 3.10.3 maintains and expands the revised cross-language benchmark methodology designed to maximize fixture reproducibility, parameter synchronization, and rigorous interpretation of performance differences:
> - **Byte-identical little-endian IEEE-754 binary fixtures** and shared CSV fixtures with SHA-256 verification.
> - **Synchronized workload sizes and selected ML hyperparameters** where the APIs provide equivalent controls.
> - **Three reporting tiers** separating shared workloads, library-to-library comparisons, and architecture-specific implementations.
> - **Median wall-clock measurements** for the primary comparison tables, with full sample distributions retained in JSON artifacts.
>
> The results measure library and implementation performance on the tested platform. They should not be interpreted as a general comparison of Swift versus Python runtimes.

---

## 🚀 What’s New in 3.10.3

### Compact Numeric Storage & Memory Layout (`SwiftDataFrame`)
- **Contiguous Dense Buffer Representation** — High-performance contiguous numerical buffers with separate null validity masks (`DenseColumn`), zero-allocation column views, and unified memory backing.
- **72× Faster Target Vector Extraction** — Zero-copy slice projection directly extracting contiguous vectors (`toTargetVector`) in **0.055 ms** (down from 3.959 ms in 3.10.2, **1.56× faster than NumPy**).
- **44× Faster Feature Normalization** — Direct vector bounds scaling (`MinMaxScaler`) in **0.751 ms** (down from 33.165 ms in 3.10.2, **1.05× faster than Scikit-Learn**).
- **10× Faster Feature Standardization** — Accelerate `vDSP` single-pass mean and variance standardization (`StandardScaler`) in **2.785 ms** (down from 28.393 ms in 3.10.2).
- **7.5× Faster Row Filtering** — Predicate bitmask extraction in **0.217 ms** (down from 1.624 ms in 3.10.2, **3.35× faster than Pandas**).
- **4.3× Faster Group Aggregation** — First-seen group dispatch with contiguous accumulator slots in **0.372 ms** (down from 1.593 ms in 3.10.2, **1.67× faster than Pandas**).
- **81% Memory Reduction (Peak RSS)** — In-memory tabular footprint reduced from 186.4 MiB to **34.4 MiB** (**4.3× less RAM than Python/Pandas**).

### Numerical Precision & Correctness Hardening (`SwiftStats`, `SwiftML`, `SwiftCluster`)
- **Catastrophic Cancellation Prevention in ANOVA** — Implemented blocked centered moments (`CenteredMoments.swift`) and exact Decimal ANOVA paths (`Stats+DecimalANOVA.swift`) guaranteeing numerical stability on large scale offsets ($10^9$).
- **Compensated Dot Products** — Added `DotProductAccuracy.compensated` (Kahan-Babuška-Neumaier summation) and `.exact` maintaining precision up to $10^{-16}$.
- **PCA Total Variance Normalization** — SVD explained variance ratios normalized across total fitted feature variance, producing exact 6-decimal parity with Scikit-Learn's `explained_variance_ratio_`.
- **Guarded Cosine Distance** — Explicit zero-magnitude query guards returning `0.0` instead of `NaN`.
- **Model State & Gradient Preservation** — Retained warm-start state and initial weights across regression iterations with compensated gradient accumulation.
- **Apache Parquet Conformance** — Standard page headers, dictionary encoding, and packed boolean bit unpacking conforming to the Apache Parquet specification.

---

## 📊 Side-by-Side Comparison: SwiftSci v3.10.3 vs v3.10.2 vs Python

The table below presents direct side-by-side wall-clock timings measured under identical hardware (Apple Silicon M-series arm64, macOS 15) and release compilation settings (`-O -whole-module-optimization`).

### 1. Tabular Operations & Feature Extraction (100,000 Rows)

| Operation | SwiftSci 3.10.2 | SwiftSci 3.10.3 | Python Baseline | Speedup vs 3.10.2 | 3.10.3 vs Python | Architectural Advantage in 3.10.3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **MinMaxScaler (Fit + Transform)** | `33.165 ms` | **`0.751 ms`** | `0.785 ms` (*Scikit-Learn*) | ⚡ **44.1× faster** | **SwiftSci 1.05×** | Contiguous dense buffer & SIMD bounds scaling |
| **StandardScaler (Fit + Transform)**| `28.393 ms` | **`2.785 ms`** | `0.892 ms` (*Scikit-Learn*) | ⚡ **10.2× faster** | Python 3.12× | Vectorized single-pass mean & variance gather |
| **Target Vector Extraction** | `3.959 ms` | **`0.055 ms`** | `0.086 ms` (*NumPy*) | ⚡ **72.0× faster** | **SwiftSci 1.56×** | Zero-copy contiguous memory slice projection |
| **Flat Feature Matrix Extraction** | `0.131 ms` | **`0.106 ms`** | `0.700 ms` (*NumPy*) | ⚡ **1.24× faster** | **SwiftSci 6.60×** | Cache-aligned flat float buffer layout |
| **Row Filtering (Boolean mask)** | `1.624 ms` | **`0.217 ms`** | `0.728 ms` (*Pandas*) | ⚡ **7.48× faster** | **SwiftSci 3.35×** | Hoisted predicate bitmask & parallel gather |
| **Sort Double Column** | `5.221 ms` | **`2.784 ms`** | `2.740 ms` (*NumPy*) | ⚡ **1.88× faster** | Parity (1.02×) | Adaptive radix sort with pre-cached keys |
| **CSV Read (100k rows × 5 cols)** | `12.948 ms` | **`6.795 ms`** | `10.377 ms` (*Pandas*) | ⚡ **1.91× faster** | **SwiftSci 1.53×** | Flat field metadata, parallel unquoted scan |
| **GroupBy + Aggregation (4 groups)**| `1.593 ms` | **`0.372 ms`** | `0.621 ms` (*Pandas*) | ⚡ **4.28× faster** | **SwiftSci 1.67×** | Compact group key dispatch & Kahan accumulator |
| **Peak RSS (Memory)** | `186.4 MiB` | **`34.4 MiB`** | `148.0 MiB` (*Pandas*) | ⚡ **5.4× less RAM** | **4.3× less RAM** | 81% memory reduction via compact contiguous storage |

---

### 2. Core Scientific Reductions & Statistics

| Scenario | SwiftSci 3.10.2 | SwiftSci 3.10.3 | Python Baseline | Speedup vs 3.10.2 | 3.10.3 vs Python | Scope / Implementation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mean Reduction** (1M doubles) | `0.073 ms` | **`0.071 ms`** | `0.116 ms` (*NumPy*) | ⚡ 1.03× faster | **SwiftSci 1.63×** | Accelerate `vDSP_meanvD` |
| **StdDev Reduction** (1M doubles) | `0.454 ms` | **`0.442 ms`** | `0.489 ms` (*NumPy*) | ⚡ 1.03× faster | **SwiftSci 1.11×** | Accelerate two-pass SIMD reduction |
| **Variance Reduction** (1M doubles)| `0.453 ms` | **`0.440 ms`** | `0.489 ms` (*NumPy*) | ⚡ 1.03× faster | **SwiftSci 1.11×** | Accelerate two-pass SIMD reduction |
| **Pearson Correlation** (500k pairs)| `0.803 ms` | **`0.795 ms`** | `1.202 ms` (*NumPy*) | ⚡ 1.01× faster | **SwiftSci 1.51×** | BLAS `cblas_ddot` dot product |
| **Two-Sample T-Test** (100k samples)| `0.263 ms` | **`0.261 ms`** | `0.391 ms` (*SciPy*) | ⚡ 1.01× faster | **SwiftSci 1.50×** | Welch unequal-variance t-test |
| **Spearman Correlation** (100k pairs)| `10.770 ms` | **`10.512 ms`** | `13.918 ms` (*SciPy*) | ⚡ 1.02× faster | **SwiftSci 1.32×** | Parallel rank transformation + Pearson |
| **ROC-AUC** (50k predictions) | `2.628 ms` | **`2.590 ms`** | `5.196 ms` (*Scikit-Learn*) | ⚡ 1.01× faster | **SwiftSci 2.01×** | Single-pass sorted trapezoidal integration |
| **OneHotEncoder** (50k rows) | `5.028 ms` | **`4.912 ms`** | `27.639 ms` (*Scikit-Learn*) | ⚡ 1.02× faster | **SwiftSci 5.63×** | SIMD categorical bitmask (13× less RAM) |

---

### 3. Machine Learning & Forecasting Models

| Model / Workload | SwiftSci 3.10.2 | SwiftSci 3.10.3 | Python Baseline | Speedup vs 3.10.2 | 3.10.3 vs Python | Scope / Implementation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **RandomForest Fit** (1k×4, 50 trees) | `4.224 ms` | **`2.319 ms`** | `31.181 ms` (*Scikit-Learn*) | ⚡ **1.82× faster** | **SwiftSci 13.45×** | Data-Oriented Design (DOD) tree buffers |
| **GBDT Regressor Fit** (1k×4, 50 est) | `8.962 ms` | **`4.754 ms`** | `32.347 ms` (*Scikit-Learn*) | ⚡ **1.88× faster** | **SwiftSci 6.80×** | Contiguous gradient-boosted ensemble |
| **HistGBDT Regressor** (1k×4, 256 bins)| `—` | **`1.284 ms`** | `18.520 ms` (*Scikit-Learn*) | — | **SwiftSci 14.42×** | 256-bin histogram split evaluation |
| **HistGBDT Classifier** (1k×4, 256 bins)| `—` | **`1.193 ms`** | `17.410 ms` (*Scikit-Learn*) | — | **SwiftSci 14.59×** | 256-bin histogram classification |
| **DecisionTree Regressor** (1k samples) | `—` | **`0.241 ms`** | `0.920 ms` (*Scikit-Learn*) | — | **SwiftSci 3.82×** | Fast recursive binary partitioning |
| **DecisionTree Classifier** (1k samples)| `—` | **`0.436 ms`** | `1.150 ms` (*Scikit-Learn*) | — | **SwiftSci 2.64×** | Optimized Gini impurity splitting |
| **OLS Linear Regression** (LAPACK) | `0.424 ms` | **`0.086 ms`** | `0.850 ms` (*Scikit-Learn*) | ⚡ **4.93× faster** | **SwiftSci 9.88×** | Accelerate LAPACK `dgels_` |
| **PCA fitTransform** (1k×100 $\to$ 10) | `1.132 ms` | **`0.027 ms`** | `0.758 ms` (*Scikit-Learn*) | ⚡ **41.9× faster** | **SwiftSci 28.07×** | Accelerate LAPACK SVD spectrum |
| **Holt-Winters Fit** (50k points, p=12) | `16.681 ms` | **`0.601 ms`** | `3299.056 ms` (*Statsmodels*) | ⚡ **27.8× faster** | **SwiftSci >5000×** | Native Nelder-Mead simplex optimization |
| **ARIMA(1,1,1) Fit** (50k points) | `2.343 ms` | **`0.041 ms`** | `392.746 ms` (*Statsmodels*) | ⚡ **57.1× faster** | **SwiftSci >9000×** | Native Gaussian likelihood recursion |

---

## What’s New in 3.10.2

### DataFrame Pipeline Optimization
- **High-Throughput CSV Parser** — flat field-metadata storage, lifetime-scoped mapped bytes, parallel unquoted scanning, and exact-capacity index allocation. Reduces 1M-row CSV ingestion from ~39 ms to ~14 ms (up to **2.87× faster**) with a **40% reduction in peak RSS** (310.1 MiB → 186.4 MiB).
- **Typed Numeric Filtering** — type dispatch hoisted outside the row loop, fused null predicates, and parallel gather scheduling. Accelerates numeric filtering by up to **1.92×** while fixing precision boundaries and NaN/infinity semantics.
- **Adaptive Typed Sorting** — pre-cached sort keys and an adaptive radix sort path for large `Double` vectors, yielding a **3.12× speedup** on 1M-row inputs while preserving stable ties and null placement.
- **Compact Group Key Identity** — flat first-seen group IDs and bounded integer lookup replacing per-row `String` interpolation, delivering up to an **11.07× speedup** on single integer keys and **4.35×** on two-key composite groups.
- **Compensated Reductions & Checked Sums** — Kahan-compensated floating-point sums and means for grouped aggregations; new opt-in `GroupedDataFrame.sumChecked()` for exact `Int64` totals with overflow protection.

---

## What’s New in 3.10.1

### Benchmarking
- **Benchmark Methodology v2** — deterministic shared fixtures, SHA-256 integrity verification, synchronized parameters, three-tier classification, and relative-performance reporting.
- **GBDT benchmark dimension alignment** — corrected feature/target dimension handling in [`MLBenchmarks.swift`](file:///Users/oleksiichumak/Developer/Xcode.projects/SwiftSci/SwiftSci/Benchmarks/Swift/MLBenchmarks.swift), producing valid wall-clock measurements for the $1,000 \times 4$ workload.
- **Descending null sort fix** — descending ordering is now preserved for non-null values while null elements remain at the end (`SwiftDataFrame`, PR #38).
- **Benchmark data loading** — Swift benchmark fixtures use a memory-mapped [`BenchmarkDataLoader`](file:///Users/oleksiichumak/Developer/Xcode.projects/SwiftSci/SwiftSci/Benchmarks/Swift/BenchmarkDataLoader.swift) for the shared binary inputs.

---

## Performance & Architecture Evolution

The following timeline records selected architectural changes, engine upgrades, and benchmark milestones across SwiftSci releases.

| Version | Release Focus & Architectural Milestones | Performance / Scaling Highlights | Status |
| :--- | :--- | :--- | :---: |
| **v3.0.0** | **Accelerate SIMD Foundation & LAPACK Solvers**<br>Apple Accelerate `vDSP` vectorization, LAPACK OLS `dgels_`, and initial Data-Oriented Design (DOD) decision trees. | Accelerate-backed single-pass vector reductions and hardware-accelerated linear regression established the baseline. | 🟢 Released |
| **v3.2.0** | **TaskGroup Concurrency & Vector Indexing**<br>Parallelized Random Forest bagging using Swift Concurrency `TaskGroup`; in-memory `VectorStore` cosine-similarity index. | Historical benchmark: in-memory Top-10 vector search over 5,000 vectors (128d) in **0.167 ms**. Multi-core parallel tree generation. | 🟢 Released |
| **v3.4.0** | **Out-of-Core Streaming & Zero-Copy Views**<br>`ChunkedDataFrame` partitioned streaming, zero-allocation row view structs (`df.rows`), POSIX `mmap` high-throughput CSV parser. | Historical benchmark: 100k-row iteration in **~12 ms**; 100k-row CSV read in **15.46 ms** (1.26× faster than Pandas); `toFlatFeatureMatrix` in **20.30 ms**. | 🟢 Released |
| **v3.5.0** | **Multi-Round Scientific Benchmarks & ML Speedups**<br>Statistical benchmark harness ($R \times I = 21$), SIMD categorical encoding, and optimized sub-ms error metrics. | Historical benchmarks: `OneHotEncoder` **5.10 ms** (5.03× vs Scikit-Learn, 13× less RAM); `Forecast Errors` **0.84 ms**; `ROC-AUC` **2.61 ms**; `Two-Sample T-Test` **0.285 ms**. | 🟢 Released |
| **v3.5.1** | **Pure-Swift NMS & Temporal Anomaly Detection**<br>Autonomous `NonMaximumSuppression.filter` replacing OpenCV/torchvision, Seasonal Hybrid ESD + MAD anomaly detection, analytical 95% forecast intervals. | Historical benchmarks: sub-millisecond pure-Swift NMS; `ARIMA(1,1,1)` fit **2.46 ms** and `Holt-Winters` fit **6.45 ms** showed substantially lower fit times than the tested Statsmodels baseline. | 🟢 Released |
| **v3.5.2** | **Parquet Engine, NumPy Ingestion & GBDT Quantile Loss**<br>Pure-Swift Apache Parquet engine with PyArrow/DuckDB/HuggingFace compatibility (RLE dictionary, def/rep levels, list<item>), native `.npy`/`.npz` tensor reader, SQLite auto-discovery, `GBDTLoss.quantile`. | Zero-dependency Parquet ingestion verified on HuggingFace 5.4k-row datasets; zero-copy little-endian NumPy tensor ingestion with ZIP64 Deflate; non-parametric GBDT quantile regression. | 🟢 Released |
| **v3.6.0** | **HNSW Index, 256-Bin HistGBDT, Concurrent AutoML & Metal Kernels**<br>Hierarchical Navigable Small World (HNSW) vector search, 256-bin histogram GBDT, TaskGroup concurrent cross-validation, CSR/CSC sparse matrices with Accelerate Sparse BLAS, parallel AutoARIMA, Metal MSL SIMD-group quantization kernels, multi-agent message bus, SQLite 1024 buffering, SCRAM-SHA-256. | HNSW $O(\log N)$ logarithmic search; HistGBDT $O(\text{numBins})$ split evaluation; Sparse BLAS 50× RAM reduction; 100% DocC API coverage across 1,749 public symbols. | 🟢 Released |
| **v3.6.1** | **WordNet Lexicon Expansion & Database Health Protocols**<br>Curated 120-synset core WordNet offline taxonomy across nouns, verbs, adjectives; Princeton WordNet data loader engine; async `ping()` and symmetric endpoint initializers for relational databases. | Offline WordNet taxonomy + Princeton format ingestion; sub-millisecond database `ping()` connectivity verification. | 🟢 Released |
| **v3.7.0** | **GBDT CoreML Exporter, Distribution Drift & Leakage Detection**<br>Direct in-memory Core ML `.mlmodel` / `.mlpackage` exporter for gradient boosted decision trees; distribution drift metrics (Wasserstein distance $W_1$ & PSI with adaptive quantile binning); heuristic statistical modality inference for tabular columns; bivariate correlation leakage detector for preprocessing sentry. | Zero-dependency iOS/macOS model deployment; sub-millisecond continuous and categorical drift monitoring; automated bivariate screening preventing data leakage. | 🟢 Released |
| **v3.8.0** | **SwiftAgent Omni-Module Architecture & Real-Time Streaming**<br>SwiftSciToolbox bridging all scientific modules; type-safe `AgentToolV2` with JSON schema validation; `ReActAgent` real-time event streaming; Working, SlidingWindow, and HNSW-backed `SemanticVectorMemory`; native SwiftUI `AgentDialogueController`. | Unified tool calling across statistics, ML, forecasting, SQL, vision, and NLP; sub-millisecond HNSW episodic recall; 60 FPS SwiftUI streaming integration. | 🟢 Released |
| **v3.8.1** | **Column Text Profiling & Precision Sweeps**<br>Native `profileTextColumn(_:)` with multilingual stopword filtering; in-memory SQLite handle; vDSP SIMD Naive Bayes; Holt-Winters Nelder-Mead phase correction ($R^2=0.997$); combinatorial TreeSHAP LUT; zero-allocation TS decomposition; hardware-routed LinearSVC. | Historical measurements: SQLite ingestion **0.032 ms**; Naive Bayes **0.026 ms**; Holt-Winters $R^2 = 0.997$, $\text{RMSE} = 0.350$; TreeSHAP **0.103 ms**; KMeans **11.19 ms**. | 🟢 Released |
| **v3.10.0** | **Verified Local LLM Runtime Pipeline**<br>Two-stage prefill/incremental generation loop; `SamplingConfiguration` logits pipeline; dynamic `positionOffset` in `RoPEEmbedding`; zero-copy `QuantizedTensor` Q4_0/Q8_0 with Metal MSL parity; `ChatMessage` & `ChatTemplate` (`.llama3`, `.chatML`, `.mistral`); resilient ReAct agent loops with schema validation; zero-leak SQLite C-pointer lifecycle under ASan. | RoPE numerical parity ($\Delta \le 1.03 \times 10^{-7}$); Q4_0 Metal MSL GEMV parity ($\Delta = 0.0000$ vs Float32); AutoARIMA zero-variance fast path ($< 0.01$ ms); 100% DocC API coverage. | 🟢 Released |
| **v3.10.1** | **Benchmark Methodology v2 & Correctness Fixes**<br>Deterministic shared binary fixtures (`.bin`), synchronized workloads, `BenchmarkDataLoader` memory-mapped loader, descending-null sorting fix (`SwiftDataFrame`, PR #38), self-contained programmatic parquet test, and GBDT benchmark dimension alignment. | GBDT fit: **9.092 ms** vs **32.815 ms** Scikit-Learn baseline (**3.61× relative performance**); full descending sort preserving trailing nulls; 100% byte-identical shared fixtures for cross-language benchmarks. | 🟢 Released |
| **v3.10.2** | **DataFrame Pipeline Optimization**<br>Typed numeric filtering outside row loop, adaptive Double radix sorting, flat first-seen group IDs, bounded integer lookup, Kahan-compensated group sums, `sumChecked()`, CSV flat field metadata, parallel unquoted scanning, exact-capacity allocation, decimal rounding fix (PR #39). | CSV: **~39 ms → ~14 ms** (2.87×), peak RSS **−40%**; group typed keys **11.07×**; two-key grouping **4.35×**; typed sort **3.12×**; Float64 filter **1.66×**. | 🟢 Released |
| **v3.10.3** | **Compact Numeric Storage, Compensated Precision & Benchmark Suite**<br>Contiguous `DenseColumn` layout, blocked centered moments (ANOVA), Kahan-Babuška compensated dot products, PCA total variance normalization, zero-guard cosine distance. | Target extraction **72.0× faster** (0.055 ms vs 3.96 ms); MinMaxScaler **44.1× faster** (0.75 ms vs 33.17 ms); StandardScaler **10.2× faster** (2.79 ms vs 28.39 ms); Filter **7.5× faster** (0.22 ms vs 1.62 ms); Peak RSS **186.4 MiB → 34.4 MiB** (81% RAM reduction, 4.3× less than Python). | 🟢 Current |

> [!IMPORTANT]
> Historical measurements were produced under the benchmark methodology, dependency versions, and workloads available at the time of each release. They are retained as architectural history and should not be treated as directly comparable with the v3.10.3 Methodology v2 results unless the fixture, dependency versions, workload, and measurement procedure are identical.

---

## Benchmark Methodology v2

### 1. Shared Fixtures

Cross-language benchmarks using the common fixture set operate on the same generated inputs:

- **IEEE-754 64-bit floating-point vectors** use little-endian binary representation (`.bin`).
- **Shared CSV fixtures** are generated once and consumed identically by both benchmark suites.
- **Fixture integrity** is verified across runs using SHA-256 manifests.
- Swift loads binary vectors via [`BenchmarkDataLoader`](file:///Users/oleksiichumak/Developer/Xcode.projects/SwiftSci/SwiftSci/Benchmarks/Swift/BenchmarkDataLoader.swift) with memory-mapped (`mmap`) zero-copy access where applicable.
- Shared fixtures establish identical input data; they do not imply identical internal implementations.

The deterministic fixture generator is located at:
[`Benchmarks/generate_fixtures.py`](file:///Users/oleksiichumak/Developer/Xcode.projects/SwiftSci/SwiftSci/Benchmarks/generate_fixtures.py)

---

### 2. Workload and Hyperparameter Synchronization

Where the underlying APIs expose equivalent parameters, benchmark configurations are synchronized:

- **PCA**: Both implementations perform a complete `fitTransform` workload ($1,000 \times 100 \rightarrow 10$ components), computing both the SVD projection and projection matrix.
- **Random Forest**: 50 trees, `max_depth = 4`, `criterion = "gini"` on identical $1,000 \times 4$ classification fixtures.
- **GBDT Regressor**: 50 estimators, `max_depth = 3`, `learning_rate = 0.1` on identical $1,000 \times 4$ continuous regression surfaces.
- **K-Means**: 10,000 samples $\times$ 4 features, 3 clusters, `max_iter = 50`.

> [!NOTE]
> Matching exposed hyperparameters does not guarantee identical internal algorithms. Differences in tree construction, initialization, split selection, numerical routines, memory layout, threading, and optimization strategies can remain between implementations.

---

### 3. Interpreting Cross-Language Results

SwiftSci and the Python reference stack use different implementation paths.

The measurements therefore capture the combined effect of:
- algorithm implementation;
- data structures;
- memory layout;
- vectorization;
- native numerical libraries;
- threading and concurrency;
- framework/runtime overhead;
- and hardware-specific optimizations.

The results are not measurements of CPython interpreter overhead alone.

---

### 4. Three-Tier Reporting

#### Tier 1 — Shared Workload / Apple-to-Apple
Identical fixtures, equivalent operation semantics, and matched workload sizes (Accelerate SIMD vs NumPy C, POSIX `mmap` CSV parsing, vector reductions). Implementation details may differ.

#### Tier 2 — Library-to-Library
Idiomatic ecosystem implementations of comparable scientific or machine-learning operations (SwiftSci vs Scikit-Learn, Statsmodels, or SHAP).

#### Tier 3 — Architecture & Hardware Specialization
Workloads where SwiftSci uses architecture-specific capabilities such as Apple Metal GPU shaders, Accelerate framework solvers, unified-memory data paths, packed quantized tensors, KV-cache decoding, or native SQLite C bindings. These results are architectural comparisons rather than strict implementation-equivalence benchmarks.

---

## Benchmark Results Matrix

All measurements below are release-build wall-clock timings on the benchmark platform described later in this document.

The primary comparison metric is the **median per-operation wall-clock time** across 21 samples ($3\text{ rounds} \times 7\text{ iterations}$).

Relative Performance is calculated as:
$$\text{Relative Performance} = \frac{\text{slower measured time}}{\text{faster measured time}}$$

The label (**SwiftSci** or **Python**) identifies which implementation achieved the lower measured time.

$$\text{Example: } \frac{0.119\text{ ms (NumPy)}}{0.081\text{ ms (SwiftSci)}} = 1.46\times \implies \textbf{SwiftSci 1.46×}$$

---

### 1. Shared Workload / Apple-to-Apple

| Benchmark Scenario | SwiftSci 3.10.3 | Python Baseline | Relative Performance | Scope / Implementation |
| :--- | ---: | ---: | :---: | :--- |
| **Mean Reduction** — 1M doubles | `0.071 ms` | `0.116 ms` (*NumPy*) | **SwiftSci 1.63×** | Accelerate `vDSP_meanvD` vs NumPy C implementation |
| **StdDev Reduction** — 1M doubles | `0.442 ms` | `0.489 ms` (*NumPy*) | **SwiftSci 1.11×** | Accelerate two-pass SIMD reduction vs NumPy `std` |
| **Variance Reduction** — 1M doubles | `0.440 ms` | `0.489 ms` (*NumPy*) | **SwiftSci 1.11×** | Accelerate two-pass SIMD reduction vs NumPy `var` |
| **Pearson Correlation** — 500k pairs | `0.795 ms` | `1.202 ms` (*NumPy*) | **SwiftSci 1.51×** | SIMD covariance / dot-product path |
| **Two-Sample T-Test** — 100k samples | `0.261 ms` | `0.391 ms` (*SciPy*) | **SwiftSci 1.50×** | Welch unequal-variance t-test |
| **Spearman Correlation** — 100k pairs | `10.512 ms` | `13.918 ms` (*SciPy*) | **SwiftSci 1.32×** | Parallel rank transformation + Pearson correlation |
| **CSV Read** — 100k rows | `6.795 ms` | `10.377 ms` (*Pandas*) | **SwiftSci 1.53×** | POSIX `mmap` zero-copy chunk parsing (PR #45 compact storage) |
| **CSV Stream Read** — 10k chunks | `47.489 ms` | `22.181 ms` (*Pandas*) | **Python 2.14×** | Swift iterator vs Pandas C engine |
| **Filter Rows** — 100k rows | `0.217 ms` | `0.728 ms` (*Pandas*) | **SwiftSci 3.35×** | Typed Swift predicate filtering (improved from 1.624 ms in 3.10.2) |
| **Sort Double Column** — 100k rows | `2.784 ms` | `2.740 ms` (*NumPy*) | **Parity 1.02×** | Adaptive radix / pre-cached sort (improved from 5.221 ms in 3.10.2) |
| **GroupBy + Aggregation** — 100k rows | `0.372 ms` | `0.621 ms` (*Pandas*) | **SwiftSci 1.67×** | Compact group key identity & Kahan accumulation |
| **Target Vector Extraction** — 100k rows | `0.055 ms` | `0.086 ms` (*NumPy*) | **SwiftSci 1.56×** | Zero-copy contiguous memory slice (improved from 3.959 ms in 3.10.2) |
| **Flat Feature Matrix** — 100k rows | `0.106 ms` | `0.700 ms` (*NumPy*) | **SwiftSci 6.60×** | Cache-aligned contiguous buffer (improved from 0.131 ms in 3.10.2) |
| **MinMaxScaler Fit + Transform** — 100k | `0.751 ms` | `0.785 ms` (*Scikit-Learn*) | **SwiftSci 1.05×** | Vectorized bounds scaling (improved from 33.165 ms in 3.10.2) |
| **StandardScaler Fit + Transform** — 100k | `2.785 ms` | `0.892 ms` (*Scikit-Learn*) | **Python 3.12×** | Vectorized vDSP normalization (improved from 28.393 ms in 3.10.2) |
| **DataFrame Hash Join** — 100k rows | `35.200 ms` | `0.456 ms` (*Pandas*) | **Python 77.19×** | Swift typed hash table vs Pandas C hashtable |
| **KMeans Fit** — 10k × 4, 3 clusters, 50 iters | `14.200 ms` | `6.944 ms` (*Scikit-Learn*) | **Python 2.05×** | Underflow-clamped SIMD distance vs Cython k-means |
| **ROC-AUC** — 50k predictions | `2.590 ms` | `5.196 ms` (*Scikit-Learn*) | **SwiftSci 2.01×** | Single-pass sorted trapezoidal integration |
| **OneHotEncoder fitTransform** — 50k rows | `4.912 ms` | `27.639 ms` (*Scikit-Learn*) | **SwiftSci 5.63×** | SIMD categorical bitmask transformation (13× less RAM) |

---

### 2. Library-to-Library

| Benchmark Scenario | SwiftSci 3.10.3 | Python Baseline | Relative Performance | Scope / Implementation |
| :--- | ---: | ---: | :---: | :--- |
| **RandomForest Fit** — 1k × 4, 50 trees, d=4, gini | `2.319 ms` | `31.181 ms` (*Scikit-Learn*) | **SwiftSci 13.45×** | Data-Oriented Design (DOD) tree buffers (improved from 4.224 ms) |
| **GBDT Regressor Fit** — 1k × 4, 50 estimators | `4.754 ms` | `32.347 ms` (*Scikit-Learn*) | **SwiftSci 6.80×** | Contiguous gradient-boosted ensemble (improved from 8.962 ms) |
| **HistGBDT Regressor** — 1k × 4, 256 bins | `1.284 ms` | `18.520 ms` (*Scikit-Learn*) | **SwiftSci 14.42×** | 256-bin histogram split evaluation |
| **HistGBDT Classifier** — 1k × 4, 256 bins | `1.193 ms` | `17.410 ms` (*Scikit-Learn*) | **SwiftSci 14.59×** | 256-bin histogram classification |
| **DecisionTree Regressor** — 1k samples | `0.241 ms` | `0.920 ms` (*Scikit-Learn*) | **SwiftSci 3.82×** | Fast recursive binary partitioning |
| **DecisionTree Classifier** — 1k samples | `0.436 ms` | `1.150 ms` (*Scikit-Learn*) | **SwiftSci 2.64×** | Optimized Gini impurity splitting |
| **OLS Linear Regression** — LAPACK `dgels_` | `0.086 ms` | `0.850 ms` (*Scikit-Learn*) | **SwiftSci 9.88×** | Accelerate LAPACK least squares |
| **PCA fitTransform** — 1k × 100 → 10 components | `0.027 ms` | `0.758 ms` (*Scikit-Learn*) | **SwiftSci 28.07×** | Accelerate LAPACK `dgesdd_` SVD spectrum |
| **IsolationForest Fit** — 1k × 10, 100 trees | `13.424 ms` | `37.539 ms` (*Scikit-Learn*) | **SwiftSci 2.80×** | Parallelized data-oriented outlier trees (18× less RAM) |
| **Holt-Winters Fit** — 50k points, period=12 | `0.601 ms` | `3299.056 ms` (*Statsmodels*) | **SwiftSci >5000×** | Native Nelder-Mead simplex optimization |
| **ARIMA(1,1,1) Fit** — 50k points | `0.041 ms` | `392.746 ms` (*Statsmodels*) | **SwiftSci >9000×** | Native Gaussian likelihood recursion optimization |
| **KernelSHAP Explain** — 5 features, 100 coalitions | `0.189 ms` | `0.434 ms` (*SHAP*) | **SwiftSci 2.30×** | Coalitional sampling with zero-division numerical guards |
| **TreeSHAP Explanation** — 100 samples | `0.101 ms` | `0.067 ms` (*SHAP*) | **Python 1.51×** | Precomputed lookup tables and zero-allocation backtracking |
| **NaiveBayes Fit** — 1k × 100, 3 classes | `0.047 ms` | `0.410 ms` (*Scikit-Learn*) | **SwiftSci 8.72×** | Accelerate `vDSP_dotprD` SIMD dot-products |

---

### 3. Architecture & Hardware Specialization

| Benchmark Scenario | SwiftSci 3.10.3 | Reference Baseline | Relative Performance | Scope / Implementation |
| :--- | ---: | ---: | :---: | :--- |
| **LinearSVC Fit** — 1k × 4, 100 epochs | `0.751 ms` | `0.384 ms` (*Scikit-Learn*) | **Python 1.95×** | Soft-margin Support Vector Classifier ($C=1.0$) |
| **VectorStore Cosine Search** — 5k × 128d, top 10 | `0.175 ms` | `0.027 ms` (*NumPy*) | **Python 6.49×** | Swift heap-based top-$k$ vs BLAS matrix multiplication |
| **Global Average Pooling + Dice** | `0.003 ms` | `0.021 ms` (*NumPy*) | **SwiftSci 7.17×** | Contiguous SIMD reduction (77× less RAM) |
| **SQLite DataFrame Ingestion** | `0.025 ms` | `0.443 ms` (*Pandas*) | **SwiftSci 17.90×** | Native in-memory SQLite C API (63× less RAM) |
| **SwiftLLM Incremental Decode** (RoPE + KV-Cache) | `0.125 ms` | `1.450 ms` (*Full Forward*) | **SwiftSci 11.60×** | Single-token decode with cached K/V vs full-sequence recomputation |
| **Metal Q4_0 GEMV** — 1024d | `0.015 ms` | `0.045 ms` (*Float32 Reference*) | **SwiftSci 3.00×** | Zero-copy packed Q4_0 evaluation using Metal MSL |
| **AutoARIMA Zero-Variance Guard** (1000 pts) | `< 0.01 ms` | N/A | — | Zero-variance fast path with bounded optimization |

> [!IMPORTANT]
> The SwiftLLM incremental-decode measurement is an architectural comparison. Incremental decoding with a KV cache performs attention over the cached context for each new token ($O(N)$ attention), while the reference measurement performs a full-sequence forward pass ($O(N^2)$). The two measurements quantify different execution strategies and should not be interpreted as a like-for-like implementation benchmark.

---

### 4. Large-Scale DataFrame Workloads (1,000,000 Rows)

Historical measurements recorded during the v3.10.2 and v3.10.3 DataFrame optimization suites against Python/Pandas and Kiraa follow. The Kiraa development build used here is unofficial and has known bugs. It remains an optional experimental reference, never an accuracy oracle or production baseline. These rows have not passed standardized conformance checks:

| Operation | SwiftSci 3.10.2 | SwiftSci 3.10.3 | pandas / Python | Kiraa | Relative Performance (3.10.3 vs Pandas) | Scope / Architecture |
| :--- | :---: | :---: | ---: | ---: | :---: | :--- |
| **CSV Ingestion (1M rows)** | `13.560 ms` | **`8.920 ms`** | `67.064 ms` | `9.171 ms` | **SwiftSci 7.52×** | Flat field metadata, parallel unquoted scan |
| **Float64 Filter** | `1.940 ms` | **`0.812 ms`** | `2.121 ms` | `2.830 ms` | **SwiftSci 2.61×** | Hoisted type dispatch & contiguous bitmask |
| **Native Int64 Filter** | `1.971 ms` | **`0.835 ms`** | `1.846 ms` | *Invalid* | **SwiftSci 2.21×** | Exact bitwise comparisons & cached nulls |
| **Single-Key Group Sum** | `4.663 ms` | **`1.420 ms`** | `4.896 ms` | `9.361 ms` | **SwiftSci 3.45×** | Bounded lookup table & Kahan summation |
| **Two-Key Group Sum** | `11.336 ms` | **`4.110 ms`** | `10.599 ms` | `36.498 ms` | **SwiftSci 2.58×** | Composite key identity & Kahan summation |
| **Stable Descending Sort** | `31.367 ms` | **`14.280 ms`** | `29.229 ms` | `58.971 ms` | **SwiftSci 2.05×** | Pre-cached sort keys & radix sort path |
| **Filter → Sort → Group Sum** | `17.012 ms` | **`6.850 ms`** | `20.118 ms` | `41.025 ms` | **SwiftSci 2.94×** | End-to-end zero intermediate allocation pipeline |
| **Peak RSS (Memory)** | `186.4 MiB` | **`51.4 MiB`** | `412.0 MiB` | — | **SwiftSci 8.01× less RAM** | 72% memory reduction vs SwiftSci v3.10.2 |

---

## Validation & Correctness Scorecard

Performance alone does not establish numerical or behavioral correctness. SwiftSci 3.10.3 therefore includes validation workloads covering forecast quality, supervised learning, numerical methods, NLP, local LLM execution, runtime safety, and optimization guards.

---

### 1. Time Series & Forecasting

| Task | Model | Dataset / Setting | Metrics | Validation |
| :--- | :--- | :--- | :--- | :---: |
| **Time Series Forecast** | `ExponentialSmoothing` (Holt-Winters) | Seasonal trend, horizon 24, period 12 | RMSE 0.350; MAE 0.281; MAPE 0.21%; $R^2$ 0.997 | 🟢 $R^2$ parity with reference ($R^2=0.997$); RMSE 0.350 vs reference 0.331 |
| **Time Series Forecast** | `ARIMAModel(1,1,1)` | Autoregressive trend, horizon 24 | RMSE 10.218; MAE 8.557; MAPE 5.87% | 🟢 Gaussian maximum-likelihood parameter estimation |

---

### 2. Supervised Learning

| Task | Model | Dataset / Setting | Metrics | Validation |
| :--- | :--- | :--- | :--- | :---: |
| **Linear Regression** | `LinearRegression` (LAPACK `dgels_`) | 3-feature linear surface, 80/20 split | RMSE 0.0577; MAE 0.0502; $R^2$ 0.9999 | 🟢 Least-squares solution validation |
| **Non-linear Regression** | `GradientBoostedTreesRegressor` | Synthetic non-linear surface, 80/20 split | RMSE 0.421; MAE 0.344; $R^2$ 0.9879 | 🟢 High-precision ensemble regression validation |
| **Histogram Regression** | `HistGradientBoostingRegressor` | 256-bin binned surface, 80/20 split | RMSE 0.395; MAE 0.312; $R^2$ 0.9890 | 🟢 Optimal histogram-splitting validation |
| **Binary Classification** | `RandomForestClassifier` | 2D decision boundary, 80/20 split, 30 trees | Accuracy 98.50%; $F_1$ 0.986; ROC-AUC 0.999 | 🟢 Predictive parity workload (parity with Scikit-Learn) |
| **Histogram Classification** | `HistGradientBoostingClassifier` | 256-bin classifier, 80/20 split | Accuracy 98.00%; $F_1$ 0.980 | 🟢 Classification validation |
| **Linear Classification** | `LinearSVC` (Metal GPU / Accelerate CPU) | Soft-margin classifier, $C=1.0$ | Accuracy 98.00%; $F_1$ 0.981 | 🟢 Convex margin optimization validation |
| **Logistic Regression** | `LogisticRegression` (Accelerate CPU) | Binary cross-entropy workload | Accuracy 97.50%; $F_1$ 0.976 | 🟢 Regularized likelihood convergence |
| **NLP Classification** | `NaiveBayesClassifier` | 3-class document bag-of-words | Accuracy 35.00%; Macro-$F_1$ 0.342 | 🟢 Posterior-inference implementation validation; not a predictive-quality claim |

---

### 3. Unsupervised Learning & Feature Scaling

| Task | Model | Dataset / Setting | Metrics | Validation |
| :--- | :--- | :--- | :--- | :---: |
| **Spectral Decomposition** | `PCA` (Accelerate LAPACK SVD) | 5D correlated Gaussian data $\rightarrow$ 2 PCs | EVR [0.6812, 0.2845]; total EVR 96.57% | 🟢 SVD / explained-variance spectrum validation |
| **Clustering** | `KMeans` (SIMD Underflow Clamped) | 3 synthetic Gaussian clusters, $N=600, k=3$ | Inertia (WCSS) 124.50; 3 centroids | 🟢 Objective non-increasing within the configured iteration bound |
| **Feature Standardization** | `StandardScaler` (vDSP SIMD) | Continuous 3-column matrix, $N=1000$ | Fitted: $\mu=29.84, \sigma=11.45$; post-scaled $\mu < 10^{-15}$ | 🟢 Exact zero-mean / unit-variance parity |

---

### 4. Statistics & NLP

| Task | Implementation | Dataset / Setting | Metrics | Validation |
| :--- | :--- | :--- | :--- | :---: |
| **Welch T-Test** | `Stats.tTest` (Welch's unequal var.) | Independent samples, $N_1=N_2=1000$ | $t$ 0.1425; $p$ 0.8867; $df$ 1987.2 | 🟢 Welch-Satterthwaite calculation validation |
| **ANOVA & Correlation** | `Stats.oneWayANOVA` & `pearsonCorrelation` | 3 groups ($N=3000$) / bivariate ($N=1000$) | $F$ 0.0892; Pearson $r$ 0.0211 | 🟢 Fisher-Snedecor $F$ and covariance calculation validation |
| **Sentiment Analysis** | `VADERSentimentAnalyzer` | Positive, negative, neutral English sentences | Compound: +0.8126, −0.7523, 0.0000 | 🟢 Rule-based lexicon reference validation (NLTK parity) |

---

### 5. Local LLM Runtime & Runtime Safety

| Verification Gate | Component | Test Setting | Observed Result | Validation |
| :--- | :--- | :--- | :--- | :---: |
| **Incremental Decode** | `TransformerDecoder` + RoPE + KV-cache | Synthetic Llama-3 block (incremental vs full) | RoPE max abs error $1.03 \times 10^{-7}$; learned positional error 0.00 | 🟢 Numerical parity within tolerance ($\Delta \le 1.03 \times 10^{-7}$) |
| **Metal Quantized GEMV** | `QuantizedLinear` / `gemv_q4_0` | Q4_0 packed weights vs MLX Float32 reference | Max abs error 0.0000 | 🟢 Numerical parity in tested workload ($\Delta = 0.0000$) |
| **Compiled Graph Decode** | `MLX.compile` single-token decode | Single-token eager vs compiled graph | Max abs error $< 10^{-4}$ | 🟢 Graph-output parity within tolerance |
| **AutoARIMA Guard** | `AutoARIMA` zero-variance guard | Constant series $y = [5.0, \dots, 5.0]$ | Order (0,0,0); runtime $< 0.01$ ms | 🟢 Zero-variance fast path with bounded optimization |
| **Agent Recovery** | `ReActAgent` + `StructuredAgentTool` | Malformed JSON / schema errors | Recovery 100%; crashes 0 | 🟢 Autonomous trajectory correction validated for tested cases |
| **SQLite Memory Safety** | `SQLiteConnection` + `sqlite3_close_v2` | AddressSanitizer (ASan) runtime audit | Leaks 0; buffer errors 0 | 🟢 Clean ASan result for tested lifecycle |

```bash
# Run the legacy accuracy research suite:
swift run -c release SwiftSciBenchmarks --research --suite Accuracy
```

---

## Reproducibility & Benchmark Configuration

### Platform
- **Hardware**: Apple Silicon M-series with Unified Memory Architecture (UMA).
- **Operating system**: macOS 15 arm64.
- **Swift**: Release build with `-O -whole-module-optimization`.
- **Apple frameworks**: Accelerate (`vDSP`, `BLAS`, `LAPACK`), Metal / MLX where applicable.
- **Python**: CPython 3.11.9.
- **NumPy**: `2.3.5`.
- **Pandas**: `3.0.2`.
- **Scikit-Learn**: `1.4.2`.
- **Statsmodels**: `0.14.2`.
- **SHAP**: `0.44.1`.
- **PyTorch**: `2.11.0`.

> [!NOTE]
> Exact package patch versions should be recorded in the benchmark environment and raw JSON artifacts. Performance results are specific to the tested dependency versions and hardware configuration.

### Measurement Procedure
- 3 benchmark rounds.
- 7 measured iterations per round.
- 21 measured samples per benchmark.
- 2 warmup iterations.
- Deterministic seed `seed = 42` where supported.
- **Primary reported metric**: **Median wall-clock execution time**.
- Raw benchmark artifacts retain individual samples and additional aggregate statistics, including 20% trimmed mean and 95% confidence intervals.

For very small operations, especially measurements below 0.1 ms, timer resolution and operating-system scheduling can contribute to the observed value. Such results should therefore be interpreted together with the raw sample distribution rather than as universal fixed execution times.

---

## Reproducing runs

Follow the [standardized benchmark guide](Benchmarks/README.md) for environment setup, builds and audited runs. Use `migration-smoke` for a quick migration check and `migration` for repeated measurements. Compare compatible run directories:

```bash
python3 Benchmarks/Tools/bench.py inventory
python3 Benchmarks/Tools/bench.py compare Benchmarks/Runs/baseline Benchmarks/Runs/candidate
```

To reproduce legacy research from the repository root:

```bash
swift run -c release SwiftSciBenchmarks --research \
  --rounds 3 --iterations 7 --json swift_research.json
python3 Benchmarks/Python/benchmarks.py --research \
  --rounds 3 --iterations 7 --json python_research.json
swift run -c release SwiftSciBenchmarks --research --suite Accuracy
```

`Benchmarks/Python/compare.py` is retired. Legacy JSON files cannot be compared with the standardized command or used as conformance certificates.

---

## Interpreting the Results

The benchmark suite demonstrates that SwiftSci can achieve competitive or lower measured execution times in a number of workloads while also exposing cases where established Python libraries remain faster.

Examples include:
- SwiftSci reductions outperforming the tested NumPy baselines in several numerical workloads.
- SwiftSci achieving faster CSV ingestion (up to 4.95× vs Pandas on 1M rows), lower RAM usage (−40%), and competitive or lower times on filtered and grouped pipelines, while Python/Pandas remains faster on certain hash joins and complex multi-key groupings.
- Scikit-Learn remaining faster for the tested PCA and K-Means workloads.
- SwiftSci showing lower measured fit times for the tested Random Forest, GBDT, Isolation Forest, Holt-Winters, and ARIMA configurations.
- Architecture-specific SwiftSci implementations benefiting from Accelerate, Metal, packed quantization, native SQLite bindings, and KV-cache decoding.

These results are workload-, implementation-, dependency-, and hardware-specific. They are intended to document the performance characteristics of SwiftSci 3.10.3 rather than establish a universal performance ranking between programming languages or ecosystems.
