# SwiftSci 3.10.3 historical accuracy report

Use the [standardized benchmark guide](Benchmarks/README.md) for current runs, audits and comparisons. The results below are historical research records. They are unvalidated under the standardized contracts and make no certification or production claims. Learned-model and GPU entries remain research unless a separately audited workload covers them. Preserve their original values when reproducing or discussing this report.

Historical numerical and predictive research measurements comparing **SwiftSci 3.10.3** (Swift 6, Apple Silicon Accelerate BLAS/LAPACK & Metal) against Python reference libraries (**NumPy 2.3.5**, **Scikit-Learn 1.8.0**, **SciPy 1.17.1**, **Statsmodels 0.14.6**, **NLTK 3.10.3**, **PyTorch 2.11.0**).

---

## 🔬 Numerical & Accuracy Enhancements: SwiftSci 3.10.3 vs 3.10.2

SwiftSci 3.10.3 introduces targeted numerical stability hardening across statistical inference, linear algebra, unsupervised learning, and model persistence:

| Numerical Subsystem | SwiftSci 3.10.2 Behavior | SwiftSci 3.10.3 Enhancement | Numerical Impact & Parity |
| :--- | :--- | :--- | :--- |
| **One-Way ANOVA (`SwiftStats`)** | Suffered catastrophic cancellation on large offsets ($x_i \approx 10^9$) due to naive $\sum x^2$ formulas. | Implemented blocked centered moments (`CenteredMoments.swift`) and exact Decimal ANOVA (`Stats+DecimalANOVA.swift`). | Exact $F$-statistic and $p$-value stability across extreme scale offsets ($10^9$ offset parity with SciPy). |
| **Dot Products (`SwiftStats`)** | Standard floating-point accumulation susceptible to error compounding in ill-conditioned vector spaces. | Added `DotProductAccuracy.compensated` (Kahan-Babuška-Neumaier) and `.exact` accumulation paths. | Precision maintained up to $10^{-16}$; prevents drift in high-dimensional cosine distances and covariance. |
| **PCA Subspace Variance (`SwiftCluster`)**| Explained variance ratio (EVR) was computed relative only to selected components, yielding $\sum \text{EVR} = 1.0$ artificially. | EVR strictly normalized against total fitted variance across all feature eigenvalues/singular values ($\sum \lambda_i$). | Exact 6-decimal parity with Scikit-Learn's `explained_variance_ratio_`. |
| **Cosine Similarity (`SwiftCluster`)** | Zero-magnitude query vectors caused division by zero, resulting in `NaN` distances and corrupted ranking. | Explicit guard returning `0.0` for zero-magnitude vectors. | Prevents `NaN` contamination in downstream vector clustering and retrieval pipelines. |
| **Regression State (`SwiftML`)** | User-supplied warm-start model state or initial weights were occasionally overwritten during re-fit. | Model state, coefficient priors, and bias terms are retained and refined using compensated residual gradients. | Predictable iterative convergence and safe transfer learning. |
| **Parquet Encoding (`SwiftDataFrame`)** | Custom compact page layout that differed from strict Apache Parquet specification. | Conformed page headers, dictionary encoding, and packed boolean bit unpacking to Apache Parquet specification. | 100% interoperability with PyArrow, DuckDB, and HuggingFace datasets. |

---

## 🔬 Deterministic Validation Methodology

The legacy report used a replica of `BenchmarkLCG` for pseudo-random inputs. A common generator alone does not establish matching workload semantics or validated outputs:

```math
\text{state}_{k+1} = (\text{state}_k \times 6364136223846793005 + 1442695040888963407) \pmod{2^{64}}
```

```math
\text{double} = \frac{\text{state} \gg 11}{2^{53}} \in [0, 1)
```

- **Identical Binary Inputs**: Python and Swift receive identical 64-bit IEEE 754 binary64 values for all shared floating-point fixtures.
- **Shared Input Caveat**: Shared fixtures guarantee identical input values; they do not imply bit-identical intermediate or final results when implementations use different numerical kernels, linear-algebra decomposition routines, or optimization strategies.
- **Platform**: Apple Silicon (arm64, macOS 15)
- **Swift**: 6.0 with Strict Concurrency (`Sendable`), Accelerate (`vDSP`, `LAPACK`, `BLAS`) & Metal GPU
- **Python Baseline**: CPython 3.11.9 with NumPy 2.3.5, Scikit-Learn 1.8.0, SciPy 1.17.1, Statsmodels 0.14.6, NLTK 3.10.3, PyTorch 2.11.0

### Parity Definitions
- **Bit-identical input**: The serialized binary64 input values are identical byte-for-byte.
- **Exact numerical match**: Raw output values are identical within the explicitly stated comparison rule.
- **Reported-precision match**: Displayed metrics are equal after the documented rounding.
- **Numerical parity**: Absolute or relative error remains below the stated acceptance tolerance.
- **Behavioral parity**: The implementation produces the expected, validated behavior for the tested inputs.

---

## 📊 Summary Parity Scorecard

| Domain | Tested Models / Algorithms | Parity Status | Max Observed Discrepancy |
| :--- | :--- | :---: | :---: |
| **Supervised Regression** | OLS Linear (LAPACK), GBDT, HistGBDT, DecisionTree | ✅ **Match at Reported Precision** | $\Delta R^2 \le 0.001$ |
| **Supervised Classification** | DecisionTree, HistGBDT, RandomForest, LinearSVC, LogisticRegression, NaiveBayes | ✅ **High Predictive Agreement** | $\Delta \text{Acc} \le 2.0\%$ |
| **Unsupervised Learning** | PCA (LAPACK SVD), K-Means (WCSS Inertia) | ✅ **Numerical Match at Reported Precision** | $\Delta \text{WCSS} = 0.00, \Delta \text{EVR} \le 0.0016$ |
| **Feature Preprocessing** | StandardScaler, MinMaxScaler | ✅ **Exact Match at Reported Precision** | $\Delta < 10^{-15}$ |
| **Time-Series Forecasting** | Holt-Winters Exponential Smoothing, ARIMA(1,1,1) | ✅ **High Predictive Agreement** | $R^2$ agrees at reported precision; lower RMSE on ARIMA |
| **Inferential Statistics** | Welch t, Student t, Paired t, One-Way ANOVA, Pearson r, Spearman ρ | ✅ **Exact Match at Reported Precision** | $\Delta < 10^{-7}$ across all p-values |
| **Natural Language Processing**| VADER Sentiment Lexicon (Compound, Pos, Neu, Neg) | ✅ **Consistent Polarity Classification** | Identical polarity class assignments |
| **Local LLM Decode (Gate 4)** | RoPE + KV-Cache incremental decoding vs full forward | ✅ **Numerical Parity Within Tolerance** | $\Delta \text{Logits} \le 1.03 \times 10^{-7}$ (tolerance $< 10^{-4}$) |
| **Metal Quantized GEMV (Gate 6)**| Native Metal MSL `gemv_q4_0` vs Float32 reference | ✅ **Numerical Parity in Tested Workload** | $\Delta = 0.0000$ |
| **Compiled Graph Decode (Gate 7)**| `MLX.compile` single-token decode graph | ✅ **Numerical Parity Within Tolerance** | $\Delta \text{Logits} < 10^{-4}$ |
| **AutoARIMA Safety (Gate 8)** | Zero-variance early exit & bounded optimization | ✅ **Validated Zero-Variance Guard** | Fast exit ($< 0.01$ ms), non-diverging |
| **Agent Resilience (Gate 10)**| ReAct parameter schema validation & loop resilience | ✅ **Validated Error Recovery** | 100% recovery across tested malformed-input cases |
| **Database Safety (Gate 11)** | SQLite C-pointer lifecycle & AddressSanitizer audit | ✅ **No Leaks / Pointer Errors Observed in Tested ASan Run** | 0 leaks, 0 invalid memory accesses |

---

## 📐 Detailed Benchmark Results by Domain

### 1. Supervised Regression (N = 1,000, 80/20 Train/Test Split)

Models trained on both analytical linear systems ($y = 3x_1 - 2x_2 + 1.5x_3 + 0.5 + \epsilon$) and non-linear target manifolds ($y = 2x_1 + 3\sin(x_2) + \epsilon$).

| Model | Configuration / Method | SwiftSci 3.10.2 | Python Baseline (Scikit-Learn) | Absolute Error ($\Delta$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **OLS Linear Regression** | Accelerate LAPACK `dgels_` vs `LinearRegression` | **RMSE = 0.0533**<br>MAE = 0.0452<br>**$R^2$ = 0.99988** | **RMSE = 0.0533**<br>MAE = 0.0452<br>**$R^2$ = 0.99988** | $\Delta \text{RMSE} = 0.000$<br>$\Delta R^2 = 0.0000$ | 🎯 **Exact Match at Reported Precision** |
| **GBDT Regressor** | 30 trees, depth = 4, $\eta = 0.1$ | **RMSE = 0.490**<br>**$R^2$ = 0.9851** | **RMSE = 0.490**<br>**$R^2$ = 0.9851** | $\Delta \text{RMSE} = 0.000$<br>$\Delta R^2 = 0.0000$ | 🎯 **Exact Match at Reported Precision** |
| **HistGBDT Regressor** | 30 trees, depth = 4, bins = 256/255 | **RMSE = 0.527**<br>**$R^2$ = 0.9827** | **RMSE = 0.511**<br>**$R^2$ = 0.9837** | $\Delta \text{RMSE} = 0.016$<br>$\Delta R^2 = 0.0010$ | ✅ **High Predictive Agreement** |
| **DecisionTree Regressor** | maxDepth = 5, minSamplesSplit = 2 | **RMSE = 0.872**<br>**$R^2$ = 0.9527** | **RMSE = 0.872**<br>**$R^2$ = 0.9527** | $\Delta \text{RMSE} = 0.000$<br>$\Delta R^2 = 0.0000$ | 🎯 **Exact Match at Reported Precision** |

> **Key Takeaway**: Analytical OLS and DecisionTree Regressor match Scikit-Learn to full display precision. Accelerate LAPACK QR/SVD factorization produces identical least-squares residuals at the reported precision.

---

### 2. Supervised Classification (N = 1,000, 80/20 Train/Test Split)

Evaluation on non-linear decision boundary ($0.8x_1 + 0.6x_2 > 0$) and multi-class text feature counts.

| Model | Configuration | SwiftSci 3.10.2 | Python Baseline (Scikit-Learn) | Absolute Error ($\Delta$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DecisionTree Classifier** | maxDepth = 5, Gini criterion | **Accuracy = 98.00%**<br>$F_1$ = 0.978 | **Accuracy = 98.00%**<br>$F_1$ = 0.978 | $\Delta \text{Acc} = 0.00\%$ | 🎯 **Exact Match at Reported Precision** |
| **HistGBDT Classifier** | 30 trees, depth = 4, $\eta = 0.1$ | **Accuracy = 97.00%**<br>$F_1$ = 0.968 | **Accuracy = 98.00%**<br>$F_1$ = 0.978 | $\Delta \text{Acc} = 1.00\%$ | ✅ **High Predictive Agreement** |
| **RandomForest Classifier** | 30 trees, maxDepth = 5, Gini | **Accuracy = 97.50%**<br>$F_1$ = 0.973 | **Accuracy = 97.00%**<br>$F_1$ = 0.967 | $\Delta \text{Acc} = 0.50\%$ | ✅ **High Predictive Agreement** |
| **LinearSVC** | $C = 1.0$, Hinge loss | **Accuracy = 99.50%**<br>$F_1$ = 0.995 | **Accuracy = 100.00%**<br>$F_1$ = 1.000 | $\Delta \text{Acc} = 0.50\%$ | ✅ **Near-Identical Margin** |
| **Logistic Regression** | Binary, L2 regularized | **Accuracy = 97.50%**<br>$F_1$ = 0.973 | **Accuracy = 99.50%**<br>$F_1$ = 0.995 | $\Delta \text{Acc} = 2.00\%$ | ✅ **Converged Boundary** |
| **Naive Bayes** | Multinomial, 3 classes, $\alpha = 1.0$ | **Accuracy = 35.00%**<br>Macro-$F_1$ = 0.342 | **Accuracy = 35.00%**<br>Macro-$F_1$ = 0.342 | $\Delta \text{Acc} = 0.00\%$ | 🎯 **Exact Match at Reported Precision** |

> **Note on Naive Bayes**: The 35.00% accuracy validates implementation agreement on a synthetic multi-class token distribution where the reference Scikit-Learn baseline produces identical multinomial posterior assignments. It serves as an implementation verification test rather than a claim of high predictive accuracy.

---

### 3. Unsupervised Learning & Spectral Decomposition

Dimensionality reduction and clustering tested on 5-dimensional Gaussian data and multi-cluster synthetic blobs.

| Algorithm | Metric / Output | SwiftSci 3.10.2 | Python Baseline (Scikit-Learn) | Absolute Error ($\Delta$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PCA (5D $\rightarrow$ 2D)** | Component 1 EVR<br>Component 2 EVR<br>Total Subspace Var | $\text{EVR}_1$ = **0.6204**<br>$\text{EVR}_2$ = **0.3796**<br>$\Sigma$ = 100.00% | $\text{EVR}_1$ = **0.6204**<br>$\text{EVR}_2$ = **0.3796**<br>$\Sigma$ = 100.00% | $\Delta \text{EVR}_1 = 0.0000$<br>$\Delta \text{EVR}_2 = 0.0000$ | 🎯 **Numerical Match at Reported Precision** |
| **K-Means ($k=3, N=600$)** | WCSS Inertia<br>Centroid Count | **Inertia = 394.31**<br>Centroids = 3 | **Inertia = 394.31**<br>Centroids = 3 | $\Delta \text{Inertia} = \mathbf{0.00}$ | 🎯 **Numerical Match at Reported Precision** |

> **Mathematical Note on PCA**: SwiftSci uses Accelerate LAPACK `dgesvd_` to compute the singular value decomposition $X = U \Sigma V^T$. The principal-component metrics agree with the Scikit-Learn full-SVD reference within the reported numerical tolerance.

---

### 4. Data Preprocessing & IEEE 754 Floating-Point Precision (N = 1,000)

Feature scaling operations evaluated for numerical stability and bounds preservation.

| Transformer | Metric | SwiftSci 3.10.2 | Python Baseline (Scikit-Learn) | Numerical Precision ($\Delta$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **StandardScaler** | Column 0 Mean ($\mu_0$)<br>Column 0 Std ($\sigma_0$)<br>Post-scaled Mean | $\mu_0$ = **30.6374**<br>$\sigma_0$ = **11.5337**<br>$\mu'$ = -3.21 × 10⁻¹⁷ | $\mu_0$ = **30.6374**<br>$\sigma_0$ = **11.5337**<br>$\mu'$ = -3.10 × 10⁻¹⁷ | $\Delta \mu_0 < 10^{-15}$<br>$\Delta \sigma_0 < 10^{-15}$ | 🎯 **Exact Match at Reported Precision** |
| **MinMaxScaler** | Data Bounds [$\min_0$, $\max_0$]<br>Scaled Range [$y_{\min}$, $y_{\max}$] | [10.03, 49.78]<br>[0.0000, 1.0000] | [10.03, 49.78]<br>[0.0000, 1.0000] | $\Delta \le 10^{-15}$ | 🎯 **Exact Match at Reported Precision** |

---

### 5. Inferential Statistics & Hypothesis Testing

Comparison of two independent samples ($N=1000$), paired samples, and multiple groups against **SciPy 1.17** (`scipy.stats`).

| Test / Metric | SwiftSci 3.10.2 | SciPy Reference (scipy.stats) | Difference ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Welch's Two-Sample t-test** | $t$ = **-0.0323**, $p$ = **0.9742503**<br>$df$ = 1840.0 | $t$ = **-0.0323**, $p$ = **0.9742503**<br>$df$ = 1840.0 | $\Delta < 10^{-7}$ | 🎯 **Exact Match at Reported Precision** |
| **Student's Pooled t-test** | $t$ = **-0.0323**, $p$ = **0.9742500** | $t$ = **-0.0323**, $p$ = **0.9742500** | $\Delta < 10^{-7}$ | 🎯 **Exact Match at Reported Precision** |
| **Paired t-test** | $t$ = **0.0328**, $p$ = **0.9738098** | $t$ = **0.0328**, $p$ = **0.9738098** | $\Delta < 10^{-7}$ | 🎯 **Exact Match at Reported Precision** |
| **One-Way ANOVA** | $F$ = **275.2785**, $p$ = **1.761608 × 10⁻¹¹⁰** | $F$ = **275.2785**, $p$ = **1.761608 × 10⁻¹¹⁰** | $\Delta < 10^{-7}$ | 🎯 **Exact Match at Reported Precision** |
| **Pearson Correlation ($r$)** | $r$ = **0.03513** | $r$ = **0.03513** | $\Delta < 10^{-5}$ | 🎯 **Exact Match at Reported Precision** |
| **Spearman Rank Correlation ($\rho$)** | $\rho$ = **0.03529** | $\rho$ = **0.03529** | $\Delta < 10^{-5}$ | 🎯 **Exact Match at Reported Precision** |

> **Numerical Verification**: $p$-values are computed using Accelerate vectorized incomplete beta and gamma functions. The reported $p$-values agree with the SciPy reference to the stated tolerance, including the tested extreme-tail ANOVA case.

---

### 6. Time-Series Forecasting (H = 24 Steps Horizon)

Comparison against **Statsmodels 0.14** on seasonal trend series ($N=500$, period = 12).

| Model | SwiftSci 3.10.2 | Python (Statsmodels) | Explanation & Analysis |
| :--- | :---: | :---: | :--- |
| **Holt-Winters (Additive)** | **RMSE = 0.350**<br>MAE = 0.275<br>**MAPE = 0.19%**<br>**$R^2$ = 0.997** | **RMSE = 0.331**<br>MAE = 0.288<br>**MAPE = 0.20%**<br>**$R^2$ = 0.997** | ✅ **High Predictive Agreement** — $R^2$ agrees at reported precision (0.997), while RMSE (0.350 vs 0.331) and MAE differ modestly due to numerical optimization paths |
| **ARIMA(1,1,1)** | **RMSE = 10.218**<br>MAE = 8.557<br>**MAPE = 5.87%**<br>$R^2$ = -1.555 | **RMSE = 22.158**<br>MAE = 20.400<br>**MAPE = 14.15%**<br>$R^2$ = -11.014 | 🟢 **Lower RMSE in this test workload** — SwiftSci RMSE is approximately 2.17× lower than the Statsmodels reference on this specific test series; parameters estimated via Gaussian maximum-likelihood recursion |

---

### 7. NLP: VADER Sentiment Polarity Scoring

Comparison against **NLTK 3.10** (`nltk.sentiment.vader.SentimentIntensityAnalyzer`).

| Test Sentence | Polarity Category | SwiftSci Compound Score | NLTK Compound Score | Parity Alignment |
| :--- | :---: | :---: | :---: | :---: |
| *"SwiftSci 3.10.2 is incredibly fast, robust and accurate!"* | **Positive** | **+0.4772** | **+0.4534** | ✅ Consistent Polarity Classification |
| *"The algorithm failed completely with disastrous and horrible errors."* | **Negative** | **-0.9052** | **-0.9243** | ✅ Consistent Polarity Classification |
| *"The dataset contains standard numerical observations and measurements."* | **Neutral** | **0.0000** | **0.0000** | 🎯 **Exact Match at Reported Precision** |

---

### 8. Local LLM Runtime Pipeline & Quantized Inference (`SwiftLLM`, `SwiftNLP`, `SwiftAgent`)

Numerical parity verification between incremental generation loops, full-context forward passes, and Metal MSL compute shaders:

| Verification Gate | Test Setting & Model | SwiftSci 3.10.2 Metric | Reference Baseline | Observed Discrepancy | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Gate 4: RoPE Incremental Decode** | 2-layer Llama-3 style TransformerDecoder | Incremental token generation with dynamic `positionOffset` | Full sequence forward pass (recomputing attention) | **maxAbsError = 1.03 × 10⁻⁷** | 🎯 **Numerical Parity Within Tolerance** (acceptance tolerance $< 10^{-4}$) |
| **Gate 4: Learned Positional Decode** | 2-layer TransformerDecoder with absolute embeddings | Incremental token decode via cached K/V | Full sequence forward pass | **maxAbsError = 0.0000** | 🎯 **Exact Match on Tested Block** |
| **Gate 6: Metal MSL GEMV Kernel** | `QuantizedLinear` (Q4_0 packed weights) | Native Metal MSL `gemv_q4_0` matrix-vector dot product | MLX Float32 reference dequantization | **maxAbsError = 0.0000** | 🎯 **Numerical Parity in Tested Workload** |
| **Gate 7: MLX.compile Parity** | Single-token decode graph | Compiled computation graph | Eager MLX execution | **maxAbsError < 10⁻⁴** | 🎯 **Numerical Parity Within Tolerance** |
| **Gate 8: AutoARIMA Zero-Variance** | Constant series ($y_t = 5.0, \forall t$) | Early guard exit in `< 0.01 ms` | Unconstrained optimization loop | **Order: (0,0,0)**, runtime $< 0.01$ ms | 🛡️ **Validated Zero-Variance Guard** |
| **Gate 10: ReAct Agent Resilience** | Malformed JSON tool inputs | Trajectory error feedback loop | Unhandled JSON exceptions | **100% recovery across tested cases**, 0 crashes | 🛡️ **Validated Error Recovery** |
| **Gate 11: Database Memory Safety** | `SQLiteConnection` + `sqlite3_close_v2` | AddressSanitizer (ASan) runtime audit | Raw C-pointer leaks | **0 leaks, 0 buffer errors** | 🛡️ **No Leaks / Pointer Errors Observed in Tested ASan Run** |

---

## ⚡ Runtime Performance Context: SwiftSci 3.10.3 vs 3.10.2 vs Python

These historical timing measurements are not accuracy metrics. See [PERFORMANCE.md](PERFORMANCE.md) for the complete performance report and the [standardized benchmark guide](Benchmarks/README.md) for current measurements.

Execution time on Apple Silicon arm64 (Release build, identical datasets):

| Benchmark Scenario | SwiftSci 3.10.2 | SwiftSci 3.10.3 | Python Baseline | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **OLS Linear Fit + Predict** (1000 × 3) | 0.424 ms | **0.086 ms** | ~2.8 ms (*Scikit-Learn*) | Accelerate LAPACK `dgels_` (⚡ **4.93× vs 3.10.2**) |
| **PCA SVD Decomposition** (500 × 5 $\rightarrow$ 2) | 0.521 ms | **0.027 ms** | ~3.1 ms (*Scikit-Learn*) | Accelerate LAPACK `dgesvd_` (⚡ **19.3× vs 3.10.2**) |
| **StandardScaler Fit + Transform** (1000 × 3) | 0.574 ms | **0.058 ms** | ~1.8 ms (*Scikit-Learn*) | Vectorized vDSP normalization (⚡ **9.89× vs 3.10.2**) |
| **MinMaxScaler Fit + Transform** (1000 × 3) | 0.667 ms | **0.063 ms** | ~2.1 ms (*Scikit-Learn*) | Vectorized vDSP bounds scaling (⚡ **10.6× vs 3.10.2**) |
| **K-Means Fit + Predict** ($N=600, k=3$) | 3.461 ms | **0.110 ms** | ~18.5 ms (*Scikit-Learn*) | Underflow-clamped Euclidean distance (⚡ **31.5× vs 3.10.2**) |
| **DecisionTree Regressor Fit + Predict** (1k samples) | 6.120 ms | **0.241 ms** | ~14.2 ms (*Scikit-Learn*) | Recursive binary partitioning (⚡ **25.4× vs 3.10.2**) |
| **DecisionTree Classifier Fit + Predict** (1k samples) | 6.990 ms | **0.436 ms** | ~16.5 ms (*Scikit-Learn*) | Gini impurity splitting (⚡ **16.0× vs 3.10.2**) |
| **RandomForest Classifier** (30 trees) | 27.230 ms | **2.319 ms** | ~58.0 ms (*Scikit-Learn*) | Parallelized tree ensemble (⚡ **11.7× vs 3.10.2**) |
| **HistGBDT Regressor** (30 trees, 256 bins) | 49.570 ms | **1.284 ms** | ~82.0 ms (*Scikit-Learn*) | 256-bin histogram splitting (⚡ **38.6× vs 3.10.2**) |
| **HistGBDT Classifier** (30 trees, 256 bins) | — | **1.193 ms** | ~74.0 ms (*Scikit-Learn*) | 256-bin histogram classification |
| **GBDT Regressor** (30 trees, depth = 4) | 89.978 ms | **4.754 ms** | ~188.0 ms (*Scikit-Learn*) | Gradient boosting on residual surface (⚡ **18.9× vs 3.10.2**) |
| **Holt-Winters Fit + Forecast** ($N=500, h=24$) | 5.676 ms | **0.601 ms** | ~12.0 ms (*Statsmodels*) | Native Nelder-Mead optimization (⚡ **9.44× vs 3.10.2**) |
| **ARIMA(1,1,1) Fit + Forecast** ($N=500, h=24$) | 1.241 ms | **0.041 ms** | ~213.0 ms (*Statsmodels*) | Likelihood recursion optimization (⚡ **30.3× vs 3.10.2**) |
| **Hypothesis Tests Suite** ($t$, ANOVA, $r$, $\rho$) | 2.268 ms | **0.082 ms** | ~8.4 ms (*SciPy*) | Vectorized incomplete beta/gamma functions (⚡ **27.7× vs 3.10.2**) |
| **VADER Sentiment Analysis** (3 sentences) | 0.018 ms | **0.007 ms** | ~0.45 ms (*NLTK*) | FNV-1a token hashing lexicon lookup (⚡ **2.57× vs 3.10.2**) |
| **SwiftLLM Single-Token Incremental Step** | 0.125 ms | **0.125 ms** | 1.450 ms (*Full Forward*) | Architectural comparison: single-token cached K/V attention |
| **Metal MSL gemv_q4_0 Matrix-Vector** | 0.015 ms | **0.015 ms** | 0.045 ms (*Float32 Reference*) | Zero-copy packed Q4_0 GPU evaluation |
| **AutoARIMA Zero-Variance Exit** | < 0.01 ms | **< 0.01 ms** | Unconstrained search loop | Guard validation: non-diverging fast exit on constant series |

---

## 🛡️ Numerical Stability & Engineering Practices

1. **IEEE 754 Binary64 Precision**: Core numerical paths in `SwiftStats`, `SwiftPreprocessing`, and `SwiftML` use 64-bit IEEE 754 floating-point (`Double`) arithmetic where applicable. Observed benchmark discrepancies are reported explicitly rather than assumed to be bounded by machine epsilon.

2. **Accelerate LAPACK Matrix Condition Verification**: Analytical solvers (`dgels_`, `dgesvd_`, `dposv_`) check matrix conditioning and throw typed `SwiftMLError.singularMatrix` or fall back to general eigensolvers for ill-conditioned systems.

3. **vDSP Vectorized Operations**: `vDSP` provides vectorized implementations of reductions, dot products, and Euclidean norms; numerical behavior is validated against the corresponding reference workloads.

4. **Swift 6 Strict Concurrency**: Swift 6 strict-concurrency checks provide compile-time enforcement for `Sendable`-based isolation across estimator and transformer types where applicable.

---

## Reproducing accuracy checks

For current conformance evidence, follow the [standardized benchmark guide](Benchmarks/README.md). The `certification` profile checks all 27 NIST cases per engine; migration workloads have separate exact or tolerance-based output checks. Certificates describe only their recorded cases.

The following commands reproduce unvalidated legacy research. They cannot establish production readiness or issue conformance certificates.

### Swift (Native Harness):
```bash
swift run -c release SwiftSciBenchmarks --research --suite Accuracy
```

### Python legacy research
```bash
python3 Benchmarks/Python/accuracy_benchmarks.py --research
```

To export machine-readable JSON:
```bash
python3 Benchmarks/Python/accuracy_benchmarks.py --research --json accuracy_results.json
```
