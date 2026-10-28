# SwiftSci 3.12.0 Accuracy & Validation Report

Use the [standardized benchmark guide](Benchmarks/README.md) for current runs, audits and comparisons. The results below include audited accuracy records for SwiftSci 3.12.0 release builds (`Benchmarks/Runs/standard-3.12.0` and `Benchmarks/Runs/migration-3.12.0`) alongside historical research records. Preserve their original values when reproducing or discussing this report.

Numerical, predictive and statistical parity measurements comparing **SwiftSci 3.12.0** (Swift 6, Apple Silicon Accelerate BLAS/LAPACK & Metal) against Python reference libraries (**NumPy 2.5.3**, **Pandas 3.0.6**, **Scikit-Learn 1.8.0**, **SciPy 1.18.1**, **Statsmodels 0.14.6**, **NLTK 3.10.3**, **SHAP 0.46.0**, **PyTorch 2.11.0**).

---

## 🔬 Numerical, Algorithmic & Security Enhancements in SwiftSci 3.12.0 (PR #80–#102)

| Subsystem | SwiftSci 3.11.0 Behavior | SwiftSci 3.12.0 Enhancement | Mathematical & Protocol Impact |
| :--- | :--- | :--- | :--- |
| **`ProbabilityMatrix` (`SwiftML`)** | `endIndex` used `Swift.max(rows.count, 6)` and `subscript` clamped to `rows.last`, repeating rows on $0 < N < 6$. | Fixed `endIndex == rows.count` and strict bounds-checked `subscript`; added `Equatable`. | Exact collection cardinality across all batch sizes $N \ge 0$ (PR #102). |
| **GPU Binary Cross-Entropy (`SwiftML`)** | Analytical BCE gradient at exact zero/extreme logits suffered sign/precision drift. | Corrected GPU BCE analytical gradient formulation at zero and saturated logits; encoded multiclass Core ML softmax on output axis. | Exact gradient parity at $z=0$ and faithful multiclass Core ML export (PR #90, #91). |
| **PCA SVD & Restored Variance (`SwiftCluster`)**| Restored PCA parameters did not validate `totalVariance`; SVD workspace used heuristic sizing. | Preserved native LAPACK SVD workspace query limits and validated positive finite `totalVariance` on restore. | Guaranteed explained-variance ratio consistency across persisted models (PR #89, #101). |
| **Local Text Embeddings (`SwiftNLP`)** | Used process-randomized Swift `Hasher`, producing different embeddings across runs. | Replaced `Hasher` with deterministic 64-bit FNV-1a token hashing. | Bit-identical cross-process local embedding vectors (PR #96). |
| **Class-Aware YOLOv8 NMS (`SwiftVision`)** | `YOLOv8Detector` used class-agnostic suppression, dropping co-located objects of different classes. | Delegated to class-aware `NonMaximumSuppression.filter` (`b1.classIndex == b2.classIndex`). | Preserves overlapping detections of distinct object classes (PR #102). |
| **`HNSWIndex` Dimensionality (`SwiftCluster`)**| Distance helpers used `min(a.count, b.count)`, silently truncating mismatched vectors. | Added `dimension: Int?` invariant and throwing `addValidated` / `searchValidated` (`HNSWError`). | Rejects empty and dimension-mismatched vectors without silent truncation (PR #102). |
| **Correlation & Metrics (`SwiftStats`, `SwiftForecast`, `SwiftVisualization`)** | `spearmanCorrelation` lacked `requireNoNaN`; `plotCorrelationHeatmap` dropped NaNs independently per column; `plotROCCurve` was order-dependent on tied scores. | Added `requireNoNaN` to Spearman, finite non-negative `epsilon` checks to `TimeSeriesMetrics` (`WAPE`/`MAPE`/`SMAPE`), pairwise-complete heatmap filtering, and tied-score grouping in ROC curves. | Order-invariant ROC curves, aligned pairwise correlations, and strict NaN/Inf rejection (PR #80, #102). |
| **Wire-Protocol & SQLite Security (`SwiftDatabase`, `SwiftAgent`)** | `SSLMode.require` connected over plaintext; MySQL handshake lacked `CLIENT_CONNECT_WITH_DB`; SQLite reused read-write handles. | Fail-closed `SSLMode.require`, dynamic `HandshakeResponse41` capability flags + `mysql_native_password` SHA-1 + `OK_Packet` validation, isolated `SQLITE_OPEN_READONLY` handle (`readOnlyDbBox`), and tokenized SQL guard in `SafeDatabaseQueryTool`. | Protocol-compliant MySQL authentication and C-API enforced read-only SQLite isolation (PR #102). |

---

## 🔬 Numerical & Accuracy Enhancements: SwiftSci 3.11.0 vs 3.10.3 (Audits 1–15)

SwiftSci 3.11.0 completes a comprehensive 15-audit cycle across the entire mathematical, algorithmic, and data-science stack, establishing strict mathematical guarantees and parity with Python/R gold standards:

| Mathematical Subsystem | SwiftSci 3.10.3 Behavior | SwiftSci 3.11.0 Enhancement | Mathematical Parity & Impact |
| :--- | :--- | :--- | :--- |
| **TreeSHAP (`SwiftExplain`)** | Tree attribution accumulated uncentered leaf predictions, failing the Shapley efficiency axiom ($\sum \phi_i \ne f(x) - \mathbb{E}[f(X)]$). | Rewrote recursive tree attribution to compute exact Shapley values evaluating leaf differences relative to dataset expectation $\mathbb{E}[f(X)]$. | Exact mathematical efficiency guarantee: $\sum_{i=1}^M \phi_i = f(x) - \mathbb{E}[f(X)]$ across all tree depths (PR #76). |
| **KernelSHAP (`SwiftExplain`)** | Unconstrained regression allowed Shapley attributions to drift from the total prediction delta $f(x) - f_{\text{null}}$. | Implemented Weighted Least Squares (WLS) subject to exact linear sum constraint, with closed-form analytical solutions for $M=1$ and $M=2$. | Guaranteed sum-to-delta convergence; eliminates residual bias in feature attribution rankings (PR #76). |
| **LIME (`SwiftExplain`)** | Fixed exponential kernel bandwidth caused weight collapse to zero on high-dimensional spaces ($D > 20$). | Implemented adaptive neighborhood bandwidth scaling $\sigma = 0.75 \sqrt{D}$. | Preserves meaningful local sample weights without distance underflow across arbitrary feature dimensions (PR #76). |
| **ETS Smoothing (`SwiftForecast`)** | Smoothing parameters required manual guessing or coarse grid search. | Integrated Nelder-Mead downhill simplex optimization minimizing Gaussian innovations negative log-likelihood for $\alpha, \beta, \gamma, \phi$. | Automated, optimal state-space parameter estimation matching R's `forecast::ets` (PR #77). |
| **Backtester & MASE (`SwiftForecast`)** | Unbounded horizons and zero-scale naive baseline caused division-by-zero in MASE evaluation. | Enforced strict horizon validation ($h \ge 1$), window size bounds ($\ge 2$), and zero-scale fallback to MAE. | Reliable, production-safe time series cross-validation and benchmark stability (PR #77). |
| **Kalman Filter (`SwiftForecast`)** | Asymmetric floating-point rounding caused error covariance $P$ to drift from positive semi-definiteness. | Enforced covariance symmetry $P = \frac{1}{2}(P + P^T)$ and clipped negative diagonal entries after update steps. | Guaranteed numerical stability over arbitrarily long state estimation trajectories (PR #77). |
| **Decision Trees (`SwiftML`)** | Categorical entropy evaluated raw counts rather than normalized probabilities; regression pruning lacked variance criterion. | Corrected Shannon entropy formulas and implemented variance-reduction post-pruning. | Exact decision boundary parity with Scikit-Learn's `DecisionTreeClassifier` and `DecisionTreeRegressor` (PR #75). |
| **Linear Regression (`SwiftML`)** | Cholesky/QR decomposition failed on collinear or singular design matrices. | Implemented Moore-Penrose pseudo-inverse fallback via SVD and added non-finite feature checks. | Robust parameter fitting on rank-deficient and collinear data (PR #75). |
| **Target Encoding (`SwiftPreprocessing`)** | Singleton categories leaked target value directly into training features (target leakage). | Implemented empirical Bayes / $m$-estimate smoothing $S_i = \frac{n \bar{y}_i + m \mu}{n + m}$. | Prevents over-fitting on rare categories; conforms to Scikit-Learn target encoding specifications (PR #74). |
| **WordNet Taxonomy (`SwiftNLP`)** | Undirected graph distance confused with hierarchical LCA taxonomy distance; cycle vulnerabilities. | Added 3-color DFS taxonomy cycle validation, dangling hypernym detection, and $O(1)$ BFS queue for Lowest Common Ancestor (LCA). | 100% taxonomy acyclicity guarantee and accurate semantic similarity metrics (Wu-Palmer, Path) (PR #78). |
| **VADER Sentiment (`SwiftNLP`)** | Unrecognized words omitted from valence array skewing ratio; contrast rule missing. | Balanced neutral token baseline ($0.0$), canonical "but" contrast rule ($0.5\times$ pre, $1.5\times$ post), and punctuation/caps scaling. | Strict polarity classification alignment with NLTK VADER reference outputs (PR #78). |

---

### 📜 Historical Reference: SwiftSci 3.10.3 vs 3.10.2 Numerical Hardening

In the previous v3.10.3 cycle, targeted numerical stability hardening was established across statistical inference, linear algebra, unsupervised learning, and model persistence:

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

| Model | Configuration / Method | SwiftSci 3.10.3 | SwiftSci 3.11.0 | Python Baseline (Scikit-Learn) | Absolute Error ($\Delta$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **OLS Linear Regression** | Accelerate LAPACK `dgels_` vs `LinearRegression` | **RMSE = 0.0533**<br>MAE = 0.0452<br>**$R^2$ = 0.99988** | **RMSE = 0.0533**<br>MAE = 0.0452<br>**$R^2$ = 0.99988** | **RMSE = 0.0533**<br>MAE = 0.0452<br>**$R^2$ = 0.99988** | $\Delta \text{RMSE} = 0.000$<br>$\Delta R^2 = 0.0000$ | 🎯 **Exact Match at Reported Precision** |
| **GBDT Regressor** | 30 trees, depth = 4, $\eta = 0.1$ | **RMSE = 0.490**<br>**$R^2$ = 0.9851** | **RMSE = 0.490**<br>**$R^2$ = 0.9851** | **RMSE = 0.490**<br>**$R^2$ = 0.9851** | $\Delta \text{RMSE} = 0.000$<br>$\Delta R^2 = 0.0000$ | 🎯 **Exact Match at Reported Precision** |
| **HistGBDT Regressor** | 30 trees, depth = 4, bins = 256/255 | **RMSE = 0.527**<br>**$R^2$ = 0.9827** | **RMSE = 0.527**<br>**$R^2$ = 0.9827** | **RMSE = 0.511**<br>**$R^2$ = 0.9837** | $\Delta \text{RMSE} = 0.016$<br>$\Delta R^2 = 0.0010$ | ✅ **High Predictive Agreement** |
| **DecisionTree Regressor** | maxDepth = 5, minSamplesSplit = 2 | **RMSE = 0.872**<br>**$R^2$ = 0.9527** | **RMSE = 0.872**<br>**$R^2$ = 0.9527** | **RMSE = 0.872**<br>**$R^2$ = 0.9527** | $\Delta \text{RMSE} = 0.000$<br>$\Delta R^2 = 0.0000$ | 🎯 **Exact Match at Reported Precision** |

> **Key Takeaway & v3.11.0 Enhancements**: Analytical OLS and DecisionTree Regressor match Scikit-Learn to full display precision. In SwiftSci 3.11.0 (PR #75), `LinearRegression` includes automatic Moore-Penrose pseudo-inverse fallback via SVD (`dgesdd_`) for collinear or singular design matrices, and `DecisionTreeRegressor` includes variance-reduction post-pruning and non-finite feature checks.

---

### 2. Supervised Classification (N = 1,000, 80/20 Train/Test Split)

Evaluation on non-linear decision boundary ($0.8x_1 + 0.6x_2 > 0$) and multi-class text feature counts.

| Model | Configuration | SwiftSci 3.10.3 | SwiftSci 3.11.0 | Python Baseline (Scikit-Learn) | Absolute Error ($\Delta$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DecisionTree Classifier** | maxDepth = 5, Gini / Entropy criterion | **Accuracy = 98.00%**<br>$F_1$ = 0.978 | **Accuracy = 98.00%**<br>$F_1$ = 0.978 | **Accuracy = 98.00%**<br>$F_1$ = 0.978 | $\Delta \text{Acc} = 0.00\%$ | 🎯 **Exact Match at Reported Precision** |
| **HistGBDT Classifier** | 30 trees, depth = 4, $\eta = 0.1$ | **Accuracy = 97.00%**<br>$F_1$ = 0.968 | **Accuracy = 97.00%**<br>$F_1$ = 0.968 | **Accuracy = 98.00%**<br>$F_1$ = 0.978 | $\Delta \text{Acc} = 1.00\%$ | ✅ **High Predictive Agreement** |
| **RandomForest Classifier** | 30 trees, maxDepth = 5, Gini | **Accuracy = 97.50%**<br>$F_1$ = 0.973 | **Accuracy = 97.50%**<br>$F_1$ = 0.973 | **Accuracy = 97.00%**<br>$F_1$ = 0.967 | $\Delta \text{Acc} = 0.50\%$ | ✅ **High Predictive Agreement** |
| **LinearSVC** | $C = 1.0$, Hinge loss | **Accuracy = 99.50%**<br>$F_1$ = 0.995 | **Accuracy = 99.50%**<br>$F_1$ = 0.995 | **Accuracy = 100.00%**<br>$F_1$ = 1.000 | $\Delta \text{Acc} = 0.50\%$ | ✅ **Near-Identical Margin** |
| **Logistic Regression** | Binary, L2 regularized | **Accuracy = 97.50%**<br>$F_1$ = 0.973 | **Accuracy = 97.50%**<br>$F_1$ = 0.973 | **Accuracy = 99.50%**<br>$F_1$ = 0.995 | $\Delta \text{Acc} = 2.00\%$ | ✅ **Converged Boundary** |
| **Naive Bayes** | Multinomial / Complement, 3 classes, $\alpha = 1.0$ | **Accuracy = 35.00%**<br>Macro-$F_1$ = 0.342 | **Accuracy = 35.00%**<br>Macro-$F_1$ = 0.342 | **Accuracy = 35.00%**<br>Macro-$F_1$ = 0.342 | $\Delta \text{Acc} = 0.00\%$ | 🎯 **Exact Match at Reported Precision** |

> **v3.11.0 Classification Enhancements**: In 3.11.0 (Audit 12, PR #75), `DecisionTreeClassifier` corrected Shannon entropy probability calculations ($p_i \log_2 p_i$) and multi-class categorical boundary stability. In `SwiftNLP` (Audit 15, PR #78), `MultinomialNaiveBayes` and `ComplementNaiveBayes` gained full Sendable thread-safety and normalized log-likelihood scoring.

---

### 3. Unsupervised Learning & Spectral Decomposition

Dimensionality reduction and clustering tested on 5-dimensional Gaussian data and multi-cluster synthetic blobs.

| Algorithm | Metric / Output | SwiftSci 3.10.3 | SwiftSci 3.11.0 | Python Baseline (Scikit-Learn) | Absolute Error ($\Delta$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PCA (5D $\rightarrow$ 2D)** | Component 1 EVR<br>Component 2 EVR<br>Total Subspace Var | $\text{EVR}_1$ = **0.6204**<br>$\text{EVR}_2$ = **0.3796**<br>$\Sigma$ = 100.00% | $\text{EVR}_1$ = **0.6204**<br>$\text{EVR}_2$ = **0.3796**<br>$\Sigma$ = 100.00% | $\text{EVR}_1$ = **0.6204**<br>$\text{EVR}_2$ = **0.3796**<br>$\Sigma$ = 100.00% | $\Delta \text{EVR}_1 = 0.0000$<br>$\Delta \text{EVR}_2 = 0.0000$ | 🎯 **Numerical Match at Reported Precision** |
| **K-Means ($k=3, N=600$)** | WCSS Inertia<br>Centroid Count | **Inertia = 394.31**<br>Centroids = 3 | **Inertia = 394.31**<br>Centroids = 3 | **Inertia = 394.31**<br>Centroids = 3 | $\Delta \text{Inertia} = \mathbf{0.00}$ | 🎯 **Numerical Match at Reported Precision** |

> **Mathematical Note on PCA**: SwiftSci uses Accelerate LAPACK `dgesvd_` to compute the singular value decomposition $X = U \Sigma V^T$. The principal-component metrics agree with the Scikit-Learn full-SVD reference within the reported numerical tolerance.

---

### 4. Data Preprocessing & IEEE 754 Floating-Point Precision (N = 1,000)

Feature scaling operations evaluated for numerical stability, bounds preservation, and target leakage prevention.

| Transformer | Metric | SwiftSci 3.10.3 | SwiftSci 3.11.0 | Python Baseline (Scikit-Learn) | Numerical Precision ($\Delta$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **StandardScaler** | Column 0 Mean ($\mu_0$)<br>Column 0 Std ($\sigma_0$)<br>Post-scaled Mean | $\mu_0$ = **30.6374**<br>$\sigma_0$ = **11.5337**<br>$\mu'$ = -3.21 × 10⁻¹⁷ | $\mu_0$ = **30.6374**<br>$\sigma_0$ = **11.5337**<br>$\mu'$ = -3.21 × 10⁻¹⁷ | $\mu_0$ = **30.6374**<br>$\sigma_0$ = **11.5337**<br>$\mu'$ = -3.10 × 10⁻¹⁷ | $\Delta \mu_0 < 10^{-15}$<br>$\Delta \sigma_0 < 10^{-15}$ | 🎯 **Exact Match at Reported Precision** |
| **MinMaxScaler** | Data Bounds [$\min_0$, $\max_0$]<br>Scaled Range [$y_{\min}$, $y_{\max}$] | [10.03, 49.78]<br>[0.0000, 1.0000] | [10.03, 49.78]<br>[0.0000, 1.0000] | [10.03, 49.78]<br>[0.0000, 1.0000] | $\Delta \le 10^{-15}$ | 🎯 **Exact Match at Reported Precision** |
| **TargetEncoder** | Rare Singleton Category (N=1)<br>Empirical Bayes Regularization | Unsmoothed target leakage<br>($S_i = \bar{y}_i$) | **Smoothed $m$-estimate**<br>$S_i = \frac{n \bar{y}_i + m \mu}{n + m}$ | **Smoothed $m$-estimate**<br>($m=10.0$) | $\Delta = 0.0000$ | 🎯 **Exact Match with Scikit-Learn TargetEncoder** |

> **v3.11.0 Preprocessing Enhancements**: SwiftSci 3.11.0 (Audit 11, PR #71, PR #74) introduces `transform(consuming:)` zero-copy memory pipelines across all transformers and eliminates target leakage in `TargetEncoder` via empirical Bayes regularization matching Scikit-Learn `TargetEncoder(smooth="auto")`.

---

### 5. Inferential Statistics & Hypothesis Testing

Comparison of two independent samples ($N=1000$), paired samples, and multiple groups against **SciPy 1.17** (`scipy.stats`).

| Test / Metric | SwiftSci 3.10.3 | SwiftSci 3.11.0 | SciPy Reference (scipy.stats) | Difference ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Welch's Two-Sample t-test** | $t$ = **-0.0323**, $p$ = **0.9742503**<br>$df$ = 1840.0 | $t$ = **-0.0323**, $p$ = **0.9742503**<br>$df$ = 1840.0 | $t$ = **-0.0323**, $p$ = **0.9742503**<br>$df$ = 1840.0 | $\Delta < 10^{-7}$ | 🎯 **Exact Match at Reported Precision** |
| **Student's Pooled t-test** | $t$ = **-0.0323**, $p$ = **0.9742500** | $t$ = **-0.0323**, $p$ = **0.9742500** | $t$ = **-0.0323**, $p$ = **0.9742500** | $\Delta < 10^{-7}$ | 🎯 **Exact Match at Reported Precision** |
| **Paired t-test** | $t$ = **0.0328**, $p$ = **0.9738098** | $t$ = **0.0328**, $p$ = **0.9738098** | $t$ = **0.0328**, $p$ = **0.9738098** | $\Delta < 10^{-7}$ | 🎯 **Exact Match at Reported Precision** |
| **One-Way ANOVA** | $F$ = **275.2785**, $p$ = **1.761608 × 10⁻¹¹⁰** | $F$ = **275.2785**, $p$ = **1.761608 × 10⁻¹¹⁰** | $F$ = **275.2785**, $p$ = **1.761608 × 10⁻¹¹⁰** | $\Delta < 10^{-7}$ | 🎯 **Exact Match at Reported Precision** |
| **Pearson Correlation ($r$)** | $r$ = **0.03513** | $r$ = **0.03513** | $r$ = **0.03513** | $\Delta < 10^{-5}$ | 🎯 **Exact Match at Reported Precision** |
| **Spearman Rank Correlation ($\rho$)** | $\rho$ = **0.03529** | $\rho$ = **0.03529** | $\rho$ = **0.03529** | $\Delta < 10^{-5}$ | 🎯 **Exact Match at Reported Precision** |

> **Numerical Verification**: $p$-values are computed using Accelerate vectorized incomplete beta and gamma functions. The reported $p$-values agree with the SciPy reference to the stated tolerance, including the tested extreme-tail ANOVA case.

---

### 6. Time-Series Forecasting (H = 24 Steps Horizon)

Comparison against **Statsmodels 0.14** on seasonal trend series ($N=500$, period = 12).

| Model | SwiftSci 3.10.3 | SwiftSci 3.11.0 | Python (Statsmodels) | Explanation & Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Holt-Winters / ETS** | **RMSE = 0.350**<br>MAE = 0.275<br>**MAPE = 0.19%**<br>**$R^2$ = 0.997** | **RMSE = 0.342**<br>MAE = 0.271<br>**MAPE = 0.18%**<br>**$R^2$ = 0.997** | **RMSE = 0.331**<br>MAE = 0.288<br>**MAPE = 0.20%**<br>**$R^2$ = 0.997** | ✅ **High Predictive Agreement** — SwiftSci 3.11.0 (Audit 14, PR #77) integrates automated Nelder-Mead downhill simplex optimization for innovations Gaussian likelihood ($\alpha, \beta, \gamma, \phi$), improving RMSE (0.342) closer to Statsmodels (0.331). |
| **ARIMA(1,1,1)** | **RMSE = 10.218**<br>MAE = 8.557<br>**MAPE = 5.87%**<br>$R^2$ = -1.555 | **RMSE = 10.218**<br>MAE = 8.557<br>**MAPE = 5.87%**<br>$R^2$ = -1.555 | **RMSE = 22.158**<br>MAE = 20.400<br>**MAPE = 14.15%**<br>$R^2$ = -11.014 | 🟢 **Lower RMSE on this benchmark horizon** — SwiftSci RMSE is approximately 2.17× lower than the Statsmodels reference; 3.11.0 adds sliding backtester horizon bounds and zero-variance guard exits (PR #77). |

---

### 7. NLP: VADER Sentiment Polarity Scoring

Comparison against **NLTK 3.10** (`nltk.sentiment.vader.SentimentIntensityAnalyzer`).

| Test Sentence | Polarity Category | SwiftSci 3.10.3 | SwiftSci 3.11.0 | NLTK Compound Score | Parity Alignment |
| :--- | :---: | :---: | :---: | :---: | :---: |
| *"SwiftSci 3.11.0 is incredibly fast, robust and accurate!"* | **Positive** | **+0.4772** | **+0.4772** | **+0.4534** | ✅ Consistent Polarity Classification |
| *"The algorithm failed completely with disastrous and horrible errors."* | **Negative** | **-0.9052** | **-0.9052** | **-0.9243** | ✅ Consistent Polarity Classification |
| *"The dataset contains standard numerical observations and measurements."* | **Neutral** | **0.0000** | **0.0000** | **0.0000** | 🎯 **Exact Match at Reported Precision** |

> **v3.11.0 Sentiment Enhancements**: SwiftSci 3.11.0 (Audit 15, PR #78) introduces the canonical "but" contrast rule ($0.5\times$ pre, $1.5\times$ post), balanced neutral token valence allocation ($0.0$ for non-lexicon words), and punctuation/caps scaling matching NLTK VADER reference outputs.

---

### 8. Local LLM Runtime Pipeline, Explainability & Architectural Safety

Numerical parity verification between incremental generation loops, full-context forward passes, compute shaders, and mathematical audit bounds:

| Verification Gate | Test Setting & Model | SwiftSci 3.10.3 Metric | SwiftSci 3.11.0 Metric | Reference Baseline | Observed Discrepancy | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Gate 4: RoPE Incremental Decode** | 2-layer Llama-3 style TransformerDecoder | maxAbsError = 1.03 × 10⁻⁷ | **maxAbsError = 1.03 × 10⁻⁷** | Full sequence forward pass | $\Delta \le 1.03 \times 10^{-7}$ | 🎯 **Numerical Parity Within Tolerance** ($< 10^{-4}$) |
| **Gate 4: Learned Positional Decode** | 2-layer TransformerDecoder with absolute embeddings | maxAbsError = 0.0000 | **maxAbsError = 0.0000** | Full sequence forward pass | $\Delta = 0.0000$ | 🎯 **Exact Match on Tested Block** |
| **Gate 6: Metal MSL GEMV Kernel** | `QuantizedLinear` (Q4_0 packed weights) | maxAbsError = 0.0000 | **maxAbsError = 0.0000** | MLX Float32 reference dequantization | $\Delta = 0.0000$ | 🎯 **Numerical Parity in Tested Workload** |
| **Gate 7: MLX.compile Parity** | Single-token decode graph | maxAbsError < 10⁻⁴ | **maxAbsError < 10⁻⁴** | Eager MLX execution | $\Delta < 10^{-4}$ | 🎯 **Numerical Parity Within Tolerance** |
| **Gate 8: AutoARIMA Zero-Variance** | Constant series ($y_t = 5.0, \forall t$) | Exit $< 0.01$ ms | **Order: (0,0,0)**, runtime $< 0.01$ ms | Unconstrained optimization loop | Fast non-diverging exit | 🛡️ **Validated Zero-Variance Guard** |
| **Gate 10: ReAct Agent Resilience** | Malformed JSON tool inputs | 100% recovery | **100% recovery across tested cases** | Unhandled JSON exceptions | 0 crashes | 🛡️ **Validated Error Recovery** |
| **Gate 11: Database Memory Safety** | `SQLiteConnection` + `sqlite3_close_v2` | 0 leaks, 0 errors | **0 leaks, 0 buffer errors** | AddressSanitizer (ASan) runtime audit | 0 errors | 🛡️ **No Leaks / Pointer Errors in ASan** |
| **Gate 12: TreeSHAP Efficiency Axiom** | 50-tree Random Forest / GBDT | Uncentered leaf drift | **$\sum_{i=1}^M \phi_i = f(x) - \mathbb{E}[f(X)]$** | Exact Shapley efficiency equation | $\Delta = 0.0000$ | 🎯 **Exact Mathematical Guarantee (PR #76)** |
| **Gate 13: KernelSHAP Linear Constraint** | 10-feature tabular model | Unconstrained drift | **Constrained WLS regression** | Linear sum-to-delta condition | $\Delta = 0.0000$ | 🎯 **Exact Linear Parity (PR #76)** |
| **Gate 14: Kalman Filter Symmetry** | 1,000-step state estimation | Rounding drift | **$P = \frac{1}{2}(P + P^T) \ge 0$** | Positive semi-definite covariance | $\Delta = 0.0000$ | 🛡️ **Exact Covariance Symmetry (PR #77)** |
| **Gate 15: WordNet Taxonomy Acyclicity**| Graph of 10,000+ synset edges | Vulnerable to cycles | **3-Color DFS verification** | Acyclic Directed Graph (DAG) | 0 cycles allowed | 🛡️ **100% Taxonomy Acyclicity (PR #78)** |

---

## ⚡ Runtime Performance Context: SwiftSci 3.12.0 vs 3.11.0 vs 3.10.3 vs Python

These execution time benchmarks compare successive SwiftSci releases on Apple Silicon arm64 (Release builds, identical datasets). See [PERFORMANCE.md](PERFORMANCE.md) for the complete performance suite and [Benchmarks/README.md](Benchmarks/README.md) for reproducible standardized runs:

| Benchmark Scenario | SwiftSci 3.10.3 | SwiftSci 3.11.0 | SwiftSci 3.12.0 | Python Baseline | Notes & Speedup |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **OLS Linear Fit + Predict** (1000 × 3) | 0.086 ms | 0.084 ms | **0.080 ms** | ~2.8 ms (*Scikit-Learn*) | Accelerate LAPACK `dgels_` (⚡ **35× vs Python**) |
| **PCA SVD Decomposition** (500 × 5 $\rightarrow$ 2) | 0.027 ms | 0.026 ms | **0.026 ms** | ~3.1 ms (*Scikit-Learn*) | Accelerate LAPACK `dgesvd_` (⚡ **119× vs Python**) |
| **StandardScaler Fit + Transform** (1000 × 3) | 0.058 ms | 0.051 ms | **0.051 ms** | ~1.8 ms (*Scikit-Learn*) | Zero-copy consuming matrix path (⚡ **35× vs Python**) |
| **MinMaxScaler Fit + Transform** (1000 × 3) | 0.063 ms | 0.058 ms | **0.058 ms** | ~2.1 ms (*Scikit-Learn*) | Vectorized vDSP bounds scaling |
| **K-Means Fit + Predict** ($N=600, k=3$) | 0.110 ms | 0.108 ms | **0.108 ms** | ~18.5 ms (*Scikit-Learn*) | Underflow-clamped Euclidean distance |
| **DecisionTree Regressor Fit + Predict** (1k samples) | 0.241 ms | 0.238 ms | **0.238 ms** | ~14.2 ms (*Scikit-Learn*) | Recursive binary partitioning with pruning |
| **DecisionTree Classifier Fit + Predict** (1k samples) | 0.436 ms | 0.429 ms | **0.429 ms** | ~16.5 ms (*Scikit-Learn*) | Shannon entropy splitting |
| **RandomForest Classifier** (30 trees) | 2.319 ms | 2.285 ms | **2.210 ms** | ~58.0 ms (*Scikit-Learn*) | Parallelized tree ensemble |
| **HistGBDT Regressor** (30 trees, 256 bins) | 1.284 ms | 1.260 ms | **1.260 ms** | ~82.0 ms (*Scikit-Learn*) | 256-bin histogram splitting |
| **HistGBDT Classifier** (30 trees, 256 bins) | 1.193 ms | 1.175 ms | **1.175 ms** | ~74.0 ms (*Scikit-Learn*) | 256-bin histogram classification |
| **GBDT Regressor** (30 trees, depth = 4) | 4.754 ms | 4.680 ms | **4.680 ms** | ~188.0 ms (*Scikit-Learn*) | Gradient boosting on residual surface |
| **Holt-Winters Fit + Forecast** ($N=500, h=24$) | 0.601 ms | 0.582 ms | **0.582 ms** | ~12.0 ms (*Statsmodels*) | Nelder-Mead Gaussian likelihood optimization |
| **ARIMA(1,1,1) Fit + Forecast** ($N=500, h=24$) | 0.041 ms | 0.040 ms | **0.040 ms** | ~213.0 ms (*Statsmodels*) | Likelihood recursion optimization |
| **Hypothesis Tests Suite** ($t$, ANOVA, $r$, $\rho$) | 0.082 ms | 0.081 ms | **0.081 ms** | ~8.4 ms (*SciPy*) | Vectorized incomplete beta/gamma functions |
| **VADER Sentiment Analysis** (3 sentences) | 0.007 ms | 0.006 ms | **0.006 ms** | ~0.45 ms (*NLTK*) | FNV-1a token hashing lexicon lookup |
| **Medium CSV Read (100,000 rows)** | 6.80 ms | 3.76 ms | **2.20 ms** | 9.92 ms (*Pandas*) | Vectorized SIMD scan & compact `Double` (⚡ **1.70× vs 3.11.0**, **4.50× vs Pandas**) |
| **Medium CSV Stream Read (100k, 10k chunks)** | 47.49 ms | 46.80 ms | **2.21 ms** | 12.82 ms (*Pandas*) | Scoped record batch discovery (⚡ **21.2× vs 3.11.0**, **5.81× vs Pandas**) |
| **Medium Sort Double Column (100,000 rows)** | 2.78 ms | 2.75 ms | **2.14 ms** | 2.90 ms (*NumPy*) | Compact `Double` adaptive radix sort (⚡ **1.29× vs 3.11.0**, **1.35× vs Python**) |
| **Medium Standard Scale (100,000 rows)** | 2.79 ms | 1.97 ms | **1.97 ms** | 0.88 ms (*Scikit-Learn*) | Direct matrix consuming pipeline (⚡ **1.42× vs 3.10.3**) |
| **SIMD Hash Inner Join (100,000 rows)** | 11.20 ms | 2.70 ms | **2.70 ms** | 9.80 ms (*Pandas*) | Vectorized probe & contiguous buffer dispatch (⚡ **4.15× vs 3.10.3**) |
| **Preprocessed Heap Allocations (100k rows)** | 42 MiB | 16 MiB | **16 MiB** | ~35 MiB (*Pandas*) | Direct `PreparedNumericMatrix` pipeline (⚡ **2.6× heap reduction**) |
| **CSV 100k Peak Resident Memory (RSS)** | 51.4 MiB | 48.7 MiB | **32.2 MiB** | 211.6 MiB (*Pandas*) | Bounded CSV field metadata & compact `Double` (⚡ **6.57× less RAM than Pandas**) |
| **SwiftLLM Single-Token Incremental Step** | 0.125 ms | 0.121 ms | **0.121 ms** | 1.450 ms (*Full Forward*) | Architectural comparison: single-token cached K/V attention |
| **Metal MSL gemv_q4_0 Matrix-Vector** | 0.015 ms | 0.014 ms | **0.014 ms** | 0.045 ms (*Float32 Reference*) | Zero-copy packed Q4_0 GPU evaluation |
| **AutoARIMA Zero-Variance Exit** | < 0.01 ms | < 0.01 ms | **< 0.01 ms** | Unconstrained search loop | Guard validation: non-diverging fast exit on constant series |

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
