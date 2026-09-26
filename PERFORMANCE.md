# SwiftSci 3.10.1 Performance Benchmarks

Comprehensive comparative benchmark results for SwiftSci 3.10.1 Release builds compared with Python data-science libraries including NumPy, Pandas, SciPy, Scikit-Learn, Statsmodels, SHAP, PyTorch, and MLX on Apple Silicon / macOS 15 arm64.

> [!NOTE]
> **Benchmark Methodology v2**
>
> SwiftSci 3.10.1 introduces a revised cross-language benchmark methodology designed to improve fixture reproducibility, parameter synchronization, and interpretation of performance differences.
>
> - Byte-identical little-endian IEEE-754 binary fixtures and shared CSV fixtures with SHA-256 verification.
> - Synchronized workload sizes and selected ML hyperparameters where the APIs provide equivalent controls.
> - Three reporting tiers separating shared workloads, library-to-library comparisons, and architecture-specific implementations.
> - Median wall-clock measurements for the primary comparison tables, with full sample distributions retained in JSON artifacts.
>
> The results measure library and implementation performance on the tested platform. They should not be interpreted as a general comparison of Swift versus Python runtimes.

---

## What’s New in 3.10.1

### Benchmarking
- **Benchmark Methodology v2** — deterministic shared fixtures, SHA-256 integrity verification, synchronized parameters, three-tier classification, and relative-performance reporting.
- **GBDT benchmark dimension alignment** — corrected feature/target dimension handling in `MLBenchmarks.swift`, producing valid wall-clock measurements for the 1,000 × 4 workload.
- **Descending null sort fix** — descending ordering is now preserved for non-null values while null elements remain at the end (`SwiftDataFrame`, PR #38).
- **Benchmark data loading** — Swift benchmark fixtures use a memory-mapped `BenchmarkDataLoader` for the shared binary inputs.

### SwiftLLM
- **Two-stage generation pipeline** — full prompt prefill followed by single-token incremental decoding using cached key/value tensors.
- **SamplingConfiguration** — logits-level handling of repetition penalty, temperature, top-k, and top-p.
- **Incremental RoPE** — dynamic `positionOffset` support with numerical parity against full-sequence recomputation.
- **GGUF quantization** — direct retention of packed Q4_0/Q8_0 layouts without an intermediate artificial dequantization representation.
- **Chat templates** — built-in `.llama3`, `.chatML`, and `.mistral` renderers.

### SwiftAgent
- **Structured parameter validation** — `AgentParameterSchema` JSON validation for tool inputs.
- **ReAct recovery** — malformed tool calls and schema errors can be corrected without terminating the agent trajectory.

### SwiftDatabase
- **SQLite lifecycle hardening** — explicit connection closing, statement finalization, and `sqlite3_close_v2`.
- **ASan validation** — zero reported leaks and buffer errors in the tested lifecycle scenarios.

### SwiftForecast
- **AutoARIMA optimization guards** — zero-variance detection with a fast exit and bounded optimization iterations.

---

## Performance & Architecture Evolution

The following timeline records selected architectural changes and benchmark milestones across SwiftSci releases.

| Version | Release Focus & Architectural Milestones | Performance / Scaling Highlights | Status |
| :--- | :--- | :--- | :---: |
| **v3.0.0** | **Accelerate SIMD Foundation & LAPACK Solvers** — Apple Accelerate `vDSP` vectorization, LAPACK OLS `dgels_`, and initial Data-Oriented Design decision trees. | Accelerate-backed vector reductions and hardware-accelerated linear regression established the baseline. | 🟢 Released |
| **v3.2.0** | **TaskGroup Concurrency & Vector Indexing** — parallelized Random Forest bagging using Swift Concurrency `TaskGroup`; in-memory `VectorStore` cosine-similarity index. | Top-10 search over 5,000 vectors × 128 dimensions measured at 0.167 ms in the historical benchmark. | 🟢 Released |
| **v3.4.0** | **Out-of-Core Streaming & Zero-Copy Views** — `ChunkedDataFrame`, zero-allocation row views, and POSIX `mmap` CSV parsing. | Historical benchmark: 100k-row iteration ~12 ms; 100k-row CSV read 15.46 ms; toFlatFeatureMatrix 20.30 ms. | 🟢 Released |
| **v3.5.0** | **Multi-Round Scientific Benchmarks & ML Speedups** — statistical benchmark harness, SIMD categorical encoding, and optimized error metrics. | Historical benchmarks included OneHotEncoder 5.10 ms, Forecast Errors 0.84 ms, ROC-AUC 2.61 ms, and two-sample t-test 0.285 ms. | 🟢 Released |
| **v3.5.1** | **Pure-Swift NMS & Temporal Anomaly Detection** — native NMS, seasonal anomaly detection, and analytical forecast intervals. | Historical ARIMA and Holt-Winters measurements showed substantially lower fit times than the tested Statsmodels baseline. | 🟢 Released |
| **v3.5.2** | **Parquet Engine, NumPy Ingestion & GBDT Quantile Loss** — Pure-Swift Parquet engine, `.npy`/`.npz` ingestion, SQLite auto-discovery, and quantile GBDT loss. | Zero-dependency Parquet ingestion and zero-copy NumPy tensor loading added to the data pipeline. | 🟢 Released |
| **v3.6.0** | **HNSW, Histogram GBDT, Concurrent AutoML & Metal Kernels** — HNSW vector search, 256-bin HistGBDT, concurrent cross-validation, sparse matrices, Metal quantization kernels, and parallel AutoARIMA. | HNSW logarithmic-search architecture, sparse-memory reductions, and Metal-accelerated numerical kernels. | 🟢 Released |
| **v3.6.1** | **WordNet & Database Health Protocols** — offline WordNet taxonomy, Princeton-format ingestion, and asynchronous database health checks. | Sub-millisecond database `ping()` in the historical benchmark. | 🟢 Released |
| **v3.7.0** | **CoreML Export, Distribution Drift & Leakage Detection** — GBDT CoreML export, Wasserstein distance, PSI, modality inference, and preprocessing leakage detection. | Native CoreML deployment and sub-millisecond drift metrics in historical workloads. | 🟢 Released |
| **v3.8.0** | **SwiftAgent Omni-Module Architecture** — typed tools, structured schemas, ReAct streaming, semantic memory, and SwiftUI agent integration. | Unified access to statistics, ML, forecasting, SQL, vision, and NLP modules. | 🟢 Released |
| **v3.8.1** | **Column Text Profiling & Precision Sweeps** — multilingual text profiling, SQLite ingestion, SIMD Naive Bayes, Holt-Winters optimization, TreeSHAP LUTs, and hardware-routed LinearSVC. | Historical measurements included SQLite ingestion 0.032 ms, Naive Bayes 0.026 ms, and Holt-Winters $R^2 = 0.997$. | 🟢 Released |
| **v3.10.0** | **Verified Local LLM Runtime Pipeline** — KV-cache decoding, incremental RoPE, GGUF packed quantization, structured agent validation, and SQLite lifecycle hardening. | RoPE parity $\Delta \le 1.03 \times 10^{-7}$; Q4_0 Metal GEMV parity within the tested numerical tolerance; AutoARIMA zero-variance fast path. | 🟢 Released |
| **v3.10.1** | **Benchmark Methodology v2 & Correctness Fixes** — deterministic shared fixtures, synchronized workloads, benchmark data loader, descending-null sorting fix, and GBDT benchmark alignment. | GBDT fit: 9.092 ms vs 32.815 ms Scikit-Learn baseline (3.61× relative performance). | 🟢 Current |

> [!IMPORTANT]
> Historical measurements were produced under the benchmark methodology, dependency versions, and workloads available at the time of each release. They are retained as architectural history and should not be treated as directly comparable with the v3.10.1 Methodology v2 results unless the fixture, dependency versions, workload, and measurement procedure are identical.

---

## Benchmark Methodology v2

### 1. Shared Fixtures

Cross-language benchmarks using the common fixture set operate on the same generated inputs.

- IEEE-754 64-bit floating-point vectors use little-endian binary representation.
- Shared CSV fixtures are generated once and consumed by both benchmark suites.
- Fixture integrity is verified using SHA-256.
- Swift uses `BenchmarkDataLoader` with memory-mapped binary loading where applicable.
- Shared fixtures establish identical input data; they do not imply identical internal implementations.

The fixture generator is located at:
[`Benchmarks/generate_fixtures.py`](file:///Users/oleksiichumak/Developer/Xcode.projects/SwiftSci/SwiftSci/Benchmarks/generate_fixtures.py)

---

### 2. Workload and Hyperparameter Synchronization

Where the underlying APIs expose equivalent parameters, benchmark configurations are synchronized.

#### PCA
Both implementations perform a complete `fitTransform` workload:
- **Input**: 1,000 × 100
- **Components**: 10
- **Operation**: fit + transform

#### Random Forest
- **Trees**: 50
- **max_depth**: 4
- **criterion**: gini

#### GBDT Regressor
- **Samples**: 1,000
- **Features**: 4
- **Estimators**: 50
- **max_depth**: 3
- **learning_rate**: 0.1

#### K-Means
- **Samples**: 10,000
- **Features**: 4
- **Clusters**: 3
- **max_iter**: 50

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
Identical fixtures, equivalent operation semantics, and matched workload sizes. Implementation details may differ.

#### Tier 2 — Library-to-Library
Idiomatic ecosystem implementations of comparable scientific or machine-learning operations. Examples include SwiftSci vs Scikit-Learn, Statsmodels, or SHAP.

#### Tier 3 — Architecture & Hardware Specialization
Workloads where SwiftSci uses architecture-specific capabilities such as:
- Apple Metal;
- Accelerate;
- unified-memory data paths;
- packed quantized tensors;
- KV-cache decoding;
- native SQLite C bindings.

These results are architectural comparisons rather than strict implementation-equivalence benchmarks.

---

## Benchmark Results Matrix

All measurements below are release-build wall-clock timings on the benchmark platform described later in this document.

The primary comparison metric is the median per-operation wall-clock time across the configured benchmark samples.

For a relative-performance value:
$$\text{relative performance} = \frac{\text{slower time}}{\text{faster time}}$$

A value above 1.0× indicates the first-named implementation in the result description has the lower measured time.

---

### 1. Shared Workload / Apple-to-Apple

| Benchmark Scenario | SwiftSci | Python Baseline | Relative Performance | Scope / Implementation |
| :--- | ---: | ---: | :---: | :--- |
| **Mean Reduction** — 1M doubles | `0.081 ms` | `0.119 ms` (NumPy) | **SwiftSci 1.46×** | Accelerate `vDSP_meanvD` vs NumPy C implementation |
| **StdDev Reduction** — 1M doubles | `0.446 ms` | `0.504 ms` (NumPy) | **SwiftSci 1.13×** | Accelerate SIMD reduction vs NumPy |
| **Variance Reduction** — 1M doubles | `0.463 ms` | `0.486 ms` (NumPy) | **SwiftSci 1.05×** | Accelerate numerical reduction vs NumPy |
| **Pearson Correlation** — 500k pairs | `0.828 ms` | `1.209 ms` (NumPy) | **SwiftSci 1.46×** | SIMD covariance / dot-product path |
| **Two-Sample T-Test** — 100k samples | `0.263 ms` | `0.411 ms` (SciPy) | **SwiftSci 1.56×** | Welch unequal-variance t-test |
| **Spearman Correlation** — 100k pairs | `11.480 ms` | `14.230 ms` (SciPy) | **SwiftSci 1.24×** | Parallel rank transformation + Pearson correlation |
| **CSV Read** — 100k rows | `18.166 ms` | `19.534 ms` (Pandas) | **SwiftSci 1.08×** | POSIX mmap parsing vs Pandas C engine |
| **CSV Stream Read** — 10k chunks | `51.611 ms` | `22.448 ms` (Pandas) | **Python 2.30×** | Swift iterator vs Pandas C engine |
| **Filter Rows** — 100k rows | `23.014 ms` | `0.741 ms` (Pandas) | **Python 31.06×** | Typed Swift filtering vs Pandas vectorized indexing |
| **Sort Double Column** — 100k rows | `44.838 ms` | `7.486 ms` (NumPy) | **Python 5.99×** | Swift sorting vs NumPy native sort |
| **DataFrame Hash Join** — 100k rows | `35.200 ms` | `0.456 ms` (Pandas) | **Python 77.19×** | Swift typed hash table vs Pandas hashtable |
| **KMeans Fit** — 10k × 4 | `17.444 ms` | `7.309 ms` (Scikit-Learn) | **Python 2.39×** | SIMD distance calculation vs native Scikit-Learn implementation |
| **ROC-AUC** — 50k predictions | `2.651 ms` | `7.601 ms` (Scikit-Learn) | **SwiftSci 2.87×** | Sorted trapezoidal integration |
| **OneHotEncoder fitTransform** — 50k rows | `5.151 ms` | `31.513 ms` (Scikit-Learn) | **SwiftSci 6.12×** | SIMD categorical transformation |

---

### 2. Library-to-Library

| Benchmark Scenario | SwiftSci | Python Baseline | Relative Performance | Scope / Implementation |
| :--- | ---: | ---: | :---: | :--- |
| **RandomForest Fit** — 1k × 4, 50 trees | `4.133 ms` | `31.228 ms` (Scikit-Learn) | **SwiftSci 7.56×** | Data-Oriented Design tree buffers |
| **GBDT Regressor Fit** — 1k × 4, 50 estimators | `9.092 ms` | `32.815 ms` (Scikit-Learn) | **SwiftSci 3.61×** | Contiguous gradient-boosted ensemble representation |
| **PCA fitTransform** — 1k × 100 → 10 | `1.132 ms` | `0.758 ms` (Scikit-Learn) | **Python 1.49×** | Accelerate LAPACK SVD vs Python numerical stack |
| **IsolationForest Fit** — 1k × 10, 100 trees | `13.523 ms` | `38.050 ms` (Scikit-Learn) | **SwiftSci 2.81×** | Parallelized data-oriented tree construction |
| **Holt-Winters Fit** — 50k points | `16.112 ms` | `3431.444 ms` (Statsmodels) | **SwiftSci 212.97×** | Native Nelder-Mead optimization |
| **ARIMA(1,1,1) Fit** — 50k points | `2.264 ms` | `594.361 ms` (Statsmodels) | **SwiftSci 262.49×** | Native Gaussian likelihood optimization |
| **KernelSHAP Explain** — 5 features, 100 coalitions | `0.189 ms` | `0.434 ms` (SHAP) | **SwiftSci 2.30×** | Coalitional sampling with numerical guards |
| **TreeSHAP Explanation** — 100 samples | `0.101 ms` | `0.081 ms` (SHAP) | **Python 1.26×** | Precomputed lookup tables and allocation reduction |
| **NaiveBayes Fit** — 1k × 100, 3 classes | `0.027 ms` | `0.410 ms` (Scikit-Learn) | **SwiftSci 15.35×** | Accelerate-backed vector operations |

---

### 3. Architecture & Hardware Specialization

| Benchmark Scenario | SwiftSci | Reference Baseline | Relative Performance | Scope / Implementation |
| :--- | ---: | ---: | :---: | :--- |
| **LinearSVC Fit** — 1k × 4, 100 epochs | `0.404 ms` | `0.384 ms` (Scikit-Learn) | **Python 1.05×** | Metal GPU / Accelerate implementation vs LibLinear |
| **VectorStore Cosine Search** — 5k × 128d, top 10 | `0.175 ms` | `0.027 ms` (NumPy) | **Python 6.49×** | Swift heap-based top-k vs matrix multiplication |
| **Global Average Pooling + Dice** | `0.003 ms` | `0.021 ms` (NumPy) | **SwiftSci 7.17×** | Contiguous SIMD operations |
| **SQLite DataFrame Ingestion** | `0.025 ms` | `0.443 ms` (Pandas) | **SwiftSci 17.90×** | Native in-memory SQLite C API |
| **SwiftLLM Incremental Decode** | `0.125 ms` | `1.450 ms` (Full Forward) | **SwiftSci 11.60×** | Single-token decode with cached K/V vs full-sequence recomputation |
| **Metal Q4_0 GEMV** — 1024d | `0.015 ms` | `0.045 ms` (Float32 dequant reference) | **SwiftSci 3.00×** | Packed Q4_0 evaluation using Metal |
| **AutoARIMA Zero-Variance Guard** | `< 0.01 ms` | N/A | — | Zero-variance fast path with bounded optimization |

> [!IMPORTANT]
> The SwiftLLM incremental-decode measurement is an architectural comparison. Incremental decoding with a KV cache performs attention over the cached context for each new token, while the reference measurement performs a full-sequence forward pass. The two measurements therefore quantify different execution strategies and should not be interpreted as a like-for-like implementation benchmark.

> [!NOTE]
> The AutoARIMA zero-variance result is a guard-behavior measurement rather than a conventional speed comparison because there is no finite reference runtime in the baseline case.

---

## Validation & Correctness Scorecard

Performance alone does not establish numerical or behavioral correctness. SwiftSci 3.10.1 therefore includes validation workloads covering forecast quality, supervised learning, numerical methods, NLP, local LLM execution, runtime safety, and optimization guards.

---

### 1. Time Series & Forecasting

| Task | Model | Dataset / Setting | Metrics | Validation |
| :--- | :--- | :--- | :--- | :---: |
| **Time Series Forecast** | `ExponentialSmoothing` / Holt-Winters | Seasonal trend, horizon 24, period 12 | RMSE 0.350; MAE 0.281; MAPE 0.21%; $R^2$ 0.997 | 🟢 $R^2$ parity with reference; RMSE 0.350 vs reference 0.331 |
| **Time Series Forecast** | `ARIMAModel(1,1,1)` | Autoregressive trend, horizon 24 | RMSE 10.218; MAE 8.557; MAPE 5.87% | 🟢 Gaussian maximum-likelihood parameter estimation |

---

### 2. Supervised Learning

| Task | Model | Dataset / Setting | Metrics | Validation |
| :--- | :--- | :--- | :--- | :---: |
| **Linear Regression** | `LinearRegression` | 3-feature linear surface, 80/20 split | RMSE 0.0577; MAE 0.0502; $R^2$ 0.9999 | 🟢 Least-squares solution validation |
| **Non-linear Regression** | `GradientBoostedTreesRegressor` | Synthetic non-linear surface, 80/20 split | RMSE 0.421; MAE 0.344; $R^2$ 0.9879 | 🟢 Regression validation |
| **Histogram Regression** | `HistGradientBoostingRegressor` | 256-bin surface, 80/20 split | RMSE 0.395; MAE 0.312; $R^2$ 0.9890 | 🟢 Histogram-splitting validation |
| **Binary Classification** | `RandomForestClassifier` | 2D decision boundary, 80/20 split, 30 trees | Accuracy 98.50%; $F_1$ 0.986; ROC-AUC 0.999 | 🟢 Predictive parity workload |
| **Histogram Classification** | `HistGradientBoostingClassifier` | 256-bin classifier, 80/20 split | Accuracy 98.00%; $F_1$ 0.980 | 🟢 Classification validation |
| **Linear Classification** | `LinearSVC` | Soft-margin classifier, $C=1.0$ | Accuracy 98.00%; $F_1$ 0.981 | 🟢 Convex margin optimization validation |
| **Logistic Regression** | `LogisticRegression` | Binary cross-entropy workload | Accuracy 97.50%; $F_1$ 0.976 | 🟢 Regularized likelihood convergence |
| **NLP Classification** | `NaiveBayesClassifier` | 3-class document bag-of-words | Accuracy 35.00%; Macro-$F_1$ 0.342 | 🟢 Posterior-inference implementation validation; not a predictive-quality claim |

---

### 3. Unsupervised Learning & Feature Scaling

| Task | Model | Dataset / Setting | Metrics | Validation |
| :--- | :--- | :--- | :--- | :---: |
| **Spectral Decomposition** | `PCA` | 5D correlated Gaussian data $\rightarrow$ 2 PCs | EVR [0.6812, 0.2845]; total EVR 96.57% | 🟢 SVD / explained-variance validation |
| **Clustering** | `KMeans` | 3 synthetic Gaussian clusters, $N=600, k=3$ | Inertia 124.50; 3 centroids | 🟢 Objective non-increasing within the configured iteration bound |
| **Feature Standardization** | `StandardScaler` | Continuous 3-column matrix, $N=1000$ | Post-scaled mean $< 10^{-15}$ | 🟢 Zero-mean / unit-variance validation |

---

### 4. Statistics & NLP

| Task | Implementation | Dataset / Setting | Metrics | Validation |
| :--- | :--- | :--- | :--- | :---: |
| **Welch T-Test** | `Stats.tTest` | Independent samples, $N_1=N_2=1000$ | $t$ 0.1425; $p$ 0.8867; $df$ 1987.2 | 🟢 Welch-Satterthwaite calculation validation |
| **ANOVA + Correlation** | `oneWayANOVA` + `Pearson` | 3 groups / bivariate sample | $F$ 0.0892; Pearson $r$ 0.0211 | 🟢 Statistical calculation validation |
| **Sentiment Analysis** | `VADERSentimentAnalyzer` | Positive, negative, neutral English sentences | Compound: +0.8126, −0.7523, 0.0000 | 🟢 Lexicon/rule-based reference validation |

---

### 5. Local LLM Runtime & Runtime Safety

| Verification Gate | Component | Test Setting | Observed Result | Validation |
| :--- | :--- | :--- | :--- | :---: |
| **Incremental Decode** | `TransformerDecoder` + RoPE + KV-cache | Synthetic Llama-3 block | RoPE max abs error 1.03e-7; learned positional error 0.00 | 🟢 Numerical parity within tolerance |
| **Metal Quantized GEMV** | `QuantizedLinear` / `gemv_q4_0` | Q4_0 packed weights vs Float32 reference | Max abs error 0.0000 | 🟢 Numerical parity in tested workload |
| **Compiled Graph Decode** | `MLX.compile` | Single-token eager vs compiled graph | Max abs error $< 10^{-4}$ | 🟢 Graph-output parity within tolerance |
| **AutoARIMA Guard** | `AutoARIMA` | Constant series $y = [5.0, \dots]$ | Order (0,0,0); runtime $< 0.01$ ms | 🟢 Zero-variance fast path |
| **Agent Recovery** | `ReActAgent` + `StructuredAgentTool` | Malformed JSON / schema errors | Recovery 100%; crashes 0 | 🟢 Recovery behavior validated for tested cases |
| **SQLite Memory Safety** | `SQLiteConnection` | AddressSanitizer audit | Leaks 0; buffer errors 0 | 🟢 Clean ASan result for tested lifecycle |

---

## Reproducibility & Benchmark Configuration

### Platform
- **Hardware**: Apple Silicon M-series with Unified Memory Architecture.
- **Operating system**: macOS 15 arm64.
- **Swift**: Release build with `-O -whole-module-optimization`.
- **Apple frameworks**: Accelerate (`vDSP`, `BLAS`, `LAPACK`), Metal / MLX where applicable.
- **Python**: CPython 3.11.9.
- **NumPy**: 2.3.5.
- **Pandas**: 3.0.2.
- **Scikit-Learn**: 1.4.x.
- **Statsmodels**: 0.14.x.
- **SHAP**: 0.44.x.
- **PyTorch**: 2.11.x.

> [!NOTE]
> Exact package patch versions should be recorded in the benchmark environment and raw JSON artifacts. Performance results are specific to the tested dependency versions and hardware configuration.

### Measurement Procedure
- 3 benchmark rounds.
- 7 measured iterations per round.
- 21 measured samples per benchmark.
- 2 warmup iterations.
- Deterministic seed 42 where supported.
- **Primary reported metric**: median wall-clock execution time.
- Raw benchmark artifacts retain the individual samples and additional aggregate statistics, including the 20% trimmed mean and confidence-interval calculations.

For very small operations, especially measurements below 0.1 ms, timer resolution and operating-system scheduling can contribute to the observed value. Such results should therefore be interpreted together with the raw sample distribution rather than as universal fixed execution times.

---

## Reproducing the Benchmarks

```bash
# 1. Swift native benchmarks
swift run -c release SwiftSciBenchmarks \
  --rounds 3 \
  --iterations 7 \
  --json swift_results.json

# 2. Python reference benchmarks
cd Benchmarks/Python
python3 benchmarks.py \
  --rounds 3 \
  --iterations 7 \
  --json python_results.json

# 3. Generate the comparison report
python3 compare.py ../../swift_results.json python_results.json

# 4. Accuracy and correctness suite
swift run -c release SwiftSciBenchmarks --suite Accuracy
```

---

## Interpreting the Results

The benchmark suite demonstrates that SwiftSci can achieve competitive or lower measured execution times in a number of workloads while also exposing cases where established Python libraries remain faster.

Examples include:
- SwiftSci reductions outperforming the tested NumPy baselines in several numerical workloads.
- Python/Pandas remaining substantially faster for some DataFrame operations such as filtering and hash joins.
- Scikit-Learn remaining faster for the tested PCA and K-Means workloads.
- SwiftSci showing lower measured fit times for the tested Random Forest, GBDT, Isolation Forest, Holt-Winters, and ARIMA configurations.
- Architecture-specific SwiftSci implementations benefiting from Accelerate, Metal, packed quantization, native SQLite bindings, and KV-cache decoding.

These results are workload-, implementation-, dependency-, and hardware-specific. They are intended to document the performance characteristics of SwiftSci 3.10.1 rather than establish a universal performance ranking between programming languages or ecosystems.
