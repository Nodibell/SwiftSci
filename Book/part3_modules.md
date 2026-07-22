# Part III — Modules

This section details the design, internal representation, and algorithmic implementations of the primary modules comprising the SwiftAnalytics ecosystem.

---

## SwiftDataFrame

### Architecture & Storage Model
`SwiftDataFrame` is the core data manipulation engine. It is designed around a **column-oriented storage pattern**, which is optimized for analytics workloads where operations target entire columns (e.g., aggregations, filtering, calculations) rather than individual rows.

```
┌───────────────────────────────────────────────────────────────┐
│                       SWIFTDATAFRAME                          │
├───────────────────────────────────────────────────────────────┤
│ Column A (Int)    ──► [ ArrayBuffer<Int> ]                    │
│ Column B (Double) ──► [ ArrayBuffer<Double> ] (Arrow Backed)  │
│ Column C (String) ──► [ ArrayBuffer<String> ]                 │
└───────────────────────────────────────────────────────────────┘
```

The underlying memory layout is backed by **Apache Arrow**. Arrow arrays provide a standardized binary format with support for null bitmaps, ensuring that null-value tracking is handled without memory alignment penalties.

### Lazy Operations & Column Masking
Instead of rebuilding arrays for intermediate filtering and sorting operations, `SwiftDataFrame` employs **lazy masking**:
* **Filtering**: Row filters generate a boolean mask array. Subsequent operations read values by mapping index transformations through the mask rather than copying rows.
* **Sorting**: A sort operation returns a list of sorted indices (`gathered(at:)`). The original data is kept intact, and any view of the sorted DataFrame uses index-indirect reads, conserving memory bandwidth.

### File Format Support
* **CSV**: Fast native CSV reader with support for escaping, custom delimiters, and automatic type inference.
* **Arrow IPC**: Full read-write support for Arrow IPC streams, enabling zero-copy exchanges with other services.
* **Parquet (Roadmap)**: A future feature for column-level compression on disk, planned for v1.0.

---

## SwiftStats

`SwiftStats` provides numerical statistics and distribution models, utilizing Apple's **Accelerate framework** (specifically vDSP and vForce) for vector arithmetic.

### Vectorized Arithmetic
Every statistics calculation operates on contiguous buffers. Vector calculations (such as sum of squares, variance, and mean) are computed via vector instructions (such as `vDSP_sveD`, `vDSP_meanvD`) which process multiple elements per CPU clock cycle.

### Supported Functions
* **Distributions**: Normal, Student's t, Chi-Squared, and F-distributions, including PDF, CDF, and inverse CDF computations.
* **Hypothesis Testing**: Two-sample Student's t-tests, Chi-Squared goodness-of-fit tests, and One-Way ANOVA.
* **Confidence Intervals**: Vectorized computations for confidence intervals of sample means and differences.

---

## SwiftML

`SwiftML` provides supervised learning models. It supports a hybrid compute structure, executing either on the GPU (via MLX) or the CPU depending on the algorithm and dataset scale.

### MLX-Based Models
Linear Regression and Logistic Regression utilize **MLX Autodiff**:
1. Features are converted into `MLXArray` tensors.
2. A loss function (MSE for Linear, Cross-Entropy for Logistic) is defined.
3. The built-in `valueAndGradient` function is called to perform automatic differentiation.
4. Parameters are updated using gradient descent optimizers (SGD or Adam) running directly on the GPU.

### Pure Swift CPU Models
Tree-based models are executed on the CPU due to their branch-heavy execution paths:
* **Decision Trees**: Classifiers and Regressors that support Gini impurity, entropy, and Mean Squared Error (MSE) splits.
* **Random Forest**: Builds ensembles of decision trees. Training is parallelized using Swift `TaskGroup`, spawning separate concurrent tasks for bootstrap sample training.
* **Gradient Boosted Decision Trees (GBDT)**: Fits trees sequentially on residual errors, providing a native, zero-dependency alternative to XGBoost.

---

## SwiftCluster

`SwiftCluster` provides unsupervised machine learning and dimensionality reduction algorithms.

### Principal Component Analysis (PCA)
Features a dual execution path resolved at runtime:
* **CPU Path**: Leverages LAPACK SVD (`dgesvd_`) from the Accelerate framework, performing singular value decomposition on centered, column-major matrices.
* **GPU Path**: Utilizes `MLX.svd` to compute principal components on the GPU.

### K-Means
Optimized for Unified Memory:
* Distance matrix calculations are parallelized on the GPU via `MLXArray` vector operations.
* Small datasets automatically fallback to CPU execution to avoid scheduling overhead.

### DBSCAN
A CPU-based spatial clustering implementation:
* Evaluates clusters based on neighborhood density ($\epsilon$ and $MinPts$).
* Uses a Breadth-First Search (BFS) queue expansion method optimized for CPU cache hit rates.

---

## SwiftForecast

`SwiftForecast` focuses on time series modeling, forecasting, and signal processing. It executes entirely on the CPU using the Accelerate framework (vDSP + LAPACK).

### Components
* **TimeSeriesDecomposition**: Implements additive and multiplicative decomposition. Computes trend (via centered moving averages), seasonal indices, and random noise. Includes ACF/PACF calculations (PACF resolved using Yule-Walker equations via LAPACK solver `dgesv`).
* **ExponentialSmoothing**: Actor-isolated models supporting Simple Exponential Smoothing (SES), Holt's Linear Trend (DES), and Holt-Winters additive/multiplicative seasonality. Includes grid-search auto-optimization to minimize Mean Squared Error (MSE).
* **ARIMA / ARIMAX**: High-performance ARIMA(p, d, q) models. Differencing is computed via `vDSP`, parameter estimation is resolved using Conditional OLS (Hannan-Rissanen) via LAPACK `dgelsd`, and forecasts are generated recursively.
* **KalmanFilter**: Actor-isolated state-space estimation. Matrix multiplications use `cblas_dgemm`, and system state updates use LAPACK `dgesv`. Includes Rauch-Tung-Striebel (RTS) smoothing.
