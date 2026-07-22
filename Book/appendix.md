# Appendix

---

## API Reference

Below is a summary list of the primary public symbols exposed by each module:

| Module | Core Structs / Classes | Core Actors | Protocols / Enums |
| :--- | :--- | :--- | :--- |
| **SwiftDataFrame** | `DataFrame`, `Column`, `Row` | - | `DataBuffer`, `DataFrameError` |
| **SwiftStats** | `Normal`, `StudentT`, `ChiSquared`, `FDistribution` | - | `StatsError` |
| **SwiftPreprocessing** | `StandardScaler`, `MinMaxScaler`, `Imputer`, `RobustScaler`, `Pipeline` | - | `Transformer` |
| **SwiftML** | `DecisionTreeClassifier`, `DecisionTreeRegressor` | `LinearRegression`, `LogisticRegression`, `RandomForestClassifier`, `GradientBoostedTreesRegressor` | `ClassifierEstimator`, `RegressorEstimator` |
| **SwiftCluster** | `DBSCAN` | `PCA`, `KMeans` | `ClusterError` |
| **SwiftForecast** | `TimeSeriesDecomposition` | `ExponentialSmoothing`, `ARIMAModel`, `KalmanFilter` | `ForecastError` |

---

## Glossary

* **Unified Memory Architecture (UMA)**: A hardware design where the CPU and GPU share the same physical memory space, allowing for zero-copy data transactions.
* **Singular Value Decomposition (SVD)**: A matrix factorization method that factorizes a matrix into singular vectors and singular values, used as the engine for PCA.
* **Autodiff (Automatic Differentiation)**: A set of techniques to numerically evaluate the derivative of a function specified by a computer program. Used in MLX to calculate loss gradients.
* **Byte Pair Encoding (BPE)**: A subword tokenization algorithm that iteratively merges the most frequent pairs of bytes or characters in a text corpus.
* **Rauch-Tung-Striebel (RTS) Smoother**: A backward-pass algorithm that improves state estimates made by a forward Kalman Filter by utilizing future data points.
* **Strict Concurrency**: A compiler verification mode in Swift 6 that checks that all shared data is thread-safe and free from data races.

---

## Mathematical Symbols

* $\mathbf{X}$ : Feature design matrix of shape $[N \times D]$, where $N$ is the number of samples and $D$ is the number of features.
* $\mathbf{y}$ : Label or target vector of length $N$.
* $\mathbf{w}$ : Weight coefficient vector of length $D$.
* $\mathbf{\Sigma}$ : Covariance matrix of shape $[D \times D]$.
* $\epsilon$ : Density threshold radius parameter for DBSCAN clustering, or error residuals in time series regressions.
* $\alpha, \beta, \gamma$ : Exponential smoothing coefficients representing level, trend, and seasonality weights.
* $p, d, q$ : Autoregressive ($p$), differencing ($d$), and moving average ($q$) order parameters for ARIMA models.

---

## References

### Books
1. **Introduction to Linear Algebra** (5th Edition) — *Gilbert Strang*. A foundational text for matrix operations, projections, and SVD.
2. **The Elements of Statistical Learning** — *Trevor Hastie, Robert Tibshirani, Jerome Friedman*. Outlines the mathematical mechanics of Decision Trees, Random Forests, and Gradient Boosting.
3. **Forecasting: Principles and Practice** (3rd Edition) — *Rob J Hyndman, George Athanasopoulos*. Covers Exponential Smoothing, ARIMA models, and time series decompositions.

### Research Papers
1. **A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise** — *Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu* (1996). Introduces the DBSCAN algorithm.
2. **An Algorithm for Least-Squares Estimation of Nonlinear Parameters** — *Donald W. Marquardt* (1963). Provides mathematical foundations for least-squares solvers.
3. **BPE: Neural Machine Translation of Rare Words with Subword Units** — *Rico Sennrich, Barry Haddow, Alexandra Birch* (2015). Details the subword tokenization process.
