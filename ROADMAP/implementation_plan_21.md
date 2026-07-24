# SwiftSci 2.0 (Phase 21) — Implementation Plan: Evaluation Metrics, Advanced Algorithms & MLOps Infrastructure

## Executive Summary

This plan details the implementation roadmap for **SwiftSci 2.0 (Phase 21)** based on the gap analysis in `missing_metrics.md` and `swiftsci_roadmap_gaps.md`.

The goal of Phase 21 is to elevate SwiftSci from a feature-complete tabular framework into a production-grade, enterprise-ready scientific computing stack (on par with SciPy, scikit-learn, statsmodels, and Polars/Arrow).

### Primary Objectives:

1. **Core API Freeze & Deprecation Governance**: Lock down and freeze public protocols (`AnyColumn`, `SupportedType`, `Estimator`, `Transformer`, `Classifier`, `Regressor`, `MetricEvaluator`), ensuring Swift 6 `Sendable` strict concurrency compliance. Mark legacy v1.x unversioned APIs with `@available(*, deprecated, message: "...")` warnings.
2. **Unsupervised & Clustering Evaluation**: Implement `Silhouette Score`, `Inertia / WCSS`, `Calinski-Harabasz Index`, `Davies-Bouldin Index`, `Contamination Ratio`, and `ARI / NMI`.
3. **Classification & Regression Metrics**: Implement `ROC-AUC`, `PR-AUC`, `MCC`, `Log-Loss`, `Balanced Accuracy`, `F-beta Score`, `R²`, `Adjusted R²`, `MAPE`, and `Explained Variance`.
4. **Advanced Validation Folds**: Implement `StratifiedKFold`, `TimeSeriesSplit`, and `GroupKFold`.
5. **Feature Engineering & Selection**: Add `PolynomialFeatures`, time-series lag/rolling features, `VarianceThreshold`, `SelectKBest`, and `Recursive Feature Elimination (RFE)`.
6. **Advanced Models**: Implement `HistGradientBoosting`, `AutoARIMA` / `SARIMAX`, `ExponentialSmoothing`, `UMAP`, `Kaplan-Meier` & `Cox Proportional Hazards`, and Isotonic/Platt Probability Calibration.
7. **MLOps & Accelerate Acceleration**: Add CoreML and ONNX exporters, multi-threaded tree building, and Saura UI Model Leaderboard dynamic metric integration.

---

## 0. Core API Freeze & Deprecation Governance

### Objectives & Strategy

- **Freeze Core Protocols**: Lock down public protocol contracts across all modules (`AnyColumn`, `SupportedType`, `Estimator`, `Transformer`, `Classifier`, `Regressor`, `MetricEvaluator`).
- **Deprecation Annotations**: Audit legacy v1.x unversioned patterns and attach Swift deprecation warnings (`@available(*, deprecated, renamed: "...", message: "...")`) to old APIs (e.g., legacy untyped dictionary methods or un-isolated static calls).
- **Swift 6 Concurrency Compliance**: Ensure all public frozen protocols inherit `Sendable` where needed for safe execution across `Task` boundaries and concurrent dispatch loops.

---

## 1. Unsupervised Learning & Clustering Metrics (`SwiftCluster` & `SwiftStats`)

### Current State

Unsupervised models (`K-Means`, `Isolation Forest`, `DBSCAN`) currently lack evaluation metrics, resulting in `0.0000` default values in UI leaderboards when true labels are absent.

### Deliverables & Formula Specifications

1. **Silhouette Coefficient / Score** (`SilhouetteScore.swift`):
   - Computes cluster cohesion \(a(i)\) (mean intra-cluster distance) and separation \(b(i)\) (mean nearest-cluster distance).
   - Formula: \(s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}\), bounded in \([-1, 1]\).
   - Accelerated using SIMD / Accelerate vDSP pairwise distance matrix computation.
2. **Inertia / WCSS (Within-Cluster Sum of Squares)**:
   - \(\sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|^2\).
3. **Calinski-Harabasz Index (Variance Ratio Criterion)**:
   - Ratio of between-clusters dispersion to within-cluster dispersion: \(s = \frac{\text{tr}(B_k)}{\text{tr}(W_k)} \times \frac{N - k}{k - 1}\).
4. **Davies-Bouldin Index**:
   - Average similarity measure of each cluster with its most similar cluster: \(DB = \frac{1}{k}\sum_{i=1}^k \max_{j \neq i} \frac{s_i + s_j}{d(\mu_i, \mu_j)}\).
5. **Contamination Ratio & Anomaly Score Distribution**:
   - Outlier threshold quantile calculation for `IsolationForest` using normalized path lengths \(s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}\).
6. **External Cluster Metrics**:
   - `Adjusted Rand Index (ARI)` & `Normalized Mutual Information (NMI)`.

---

## 2. Supervised Classification & Regression Metrics (`SwiftOptimize` & `SwiftStats`)

### Classification Metrics (`Metrics.swift`):

1. **ROC-AUC**: Receiver Operating Characteristic Area Under Curve via trapezoidal integration over sorted probability thresholds.
2. **PR-AUC**: Area under the Precision-Recall curve for highly imbalanced datasets.
3. **Matthews Correlation Coefficient (MCC)**:
   - \(\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}\), bounded in \([-1, 1]\).
4. **Log-Loss / Cross-Entropy Loss**:
   - \(L = -\frac{1}{N}\sum_{i=1}^N [y_i \log(p_i) + (1-y_i)\log(1-p_i)]\).
5. **Balanced Accuracy**: Macro-averaged recall across all classes.
6. **F-beta Score**: Weighted harmonic mean prioritizing precision (\(\beta=0.5\)) or recall (\(\beta=2.0\)).

### Regression Metrics (`Metrics.swift`):

1. **R² Score (Coefficient of Determination)**: \(1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}\).
2. **Adjusted R² Score**: \(1 - (1 - R^2)\frac{N - 1}{N - p - 1}\).
3. **MAPE (Mean Absolute Percentage Error)**: \(\frac{100\%}{N}\sum \left|\frac{y_i - \hat{y}_i}{y_i}\right|\).
4. **Explained Variance Score**: \(1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}\).

---

## 3. Advanced Validation & Resampling (`SwiftOptimize`)

1. **`StratifiedKFold`** (`StratifiedKFold.swift`):
   - Preserves percentage of samples for each class across all folds.
2. **`TimeSeriesSplit`** (`TimeSeriesSplit.swift`):
   - Expanding-window cross-validation for sequential data to prevent look-ahead bias and data leakage.
3. **`GroupKFold`** (`GroupKFold.swift`):
   - Ensures identical group identifiers (e.g. subject/patient IDs) do not appear in both train and test splits simultaneously.

---

## 4. Advanced Preprocessing & Feature Engineering (`SwiftPreprocessing`)

1. **`PolynomialFeatures`** (`PolynomialFeatures.swift`):
   - Generates degree-\(n\) polynomial combinations and interaction features (\(x_1^2, x_1 x_2, x_2^2\)).
2. **Time-Series Lags & Rolling Aggregators** (`DataFrame+Preprocessing.swift`):
   - `withRollingMean(column:window:)`, `withRollingStd(column:window:)`, `withEWMA(column:alpha:)`, and `withLaggedColumns([Int])`.
3. **Automated Feature Selection** (`FeatureSelection.swift`):
   - `VarianceThreshold`: Removes low-variance feature vectors.
   - `SelectKBest`: Selects top \(k\) features based on ANOVA F-score or Mutual Information.
   - `RecursiveFeatureElimination (RFE)`: Iteratively prunes least significant features using model feature importances.

---

## 5. Advanced Machine Learning & Statistical Models (`SwiftML` & `SwiftForecast`)

1. **`HistGradientBoosting`** (`HistGradientBoostingRegressor.swift` / `HistGradientBoostingClassifier.swift`):
   - Integer-binned numerical feature splits (256 bins) for 10x–50x speedups on large tabular datasets.
2. **Time Series Extensions** (`SwiftForecast`):
   - `AutoARIMA` & `SARIMAX` (Seasonal ARIMA with exogenous regressors).
   - `ExponentialSmoothing` (Holt-Winters additive/multiplicative models).
   - `VAR` (Vector Autoregression).
3. **Survival Analysis** (`SwiftStats/Survival/`):
   - `KaplanMeier Estimator` (Survival curves with right-censoring).
   - `CoxProportionalHazards Model`.
4. **Probability Calibration** (`ProbabilityCalibration.swift`):
   - `IsotonicRegression` & `PlattScaling` for converting raw logits into calibrated confidence probabilities.

---

## 6. Serialization & Hardware Acceleration (`SwiftML` & `SwiftONNX`)

1. **CoreML Exporter** (`CoreMLExporter.swift`):
   - Serializes trained `SwiftML` models into Apple `.mlmodel` / `.mlpackage` bundles for Neural Engine (ANE) hardware execution.
2. **ONNX Exporter** (`ONNXExporter.swift`):
   - Exports model weights and graph specifications into ONNX format for cross-platform interop.
3. **Multi-threaded CPU Construction**:
   - `TaskGroup` parallelization for tree ensemble training and cross-validation evaluation.

---

## 7. Update DooC

- Update DooC to maintain 100% coverage
- Update docs/ for gh pages

---

## Deliverables Summary

| Component                      | Target File                                               | Purpose                                                  |
| :----------------------------- | :-------------------------------------------------------- | :------------------------------------------------------- |
| **Core Governance**      | `SwiftCore/Protocols/CoreProtocols.swift`               | Freeze public protocols & add deprecation annotations    |
| **Unsupervised Metrics** | `SwiftCluster/Metrics/SilhouetteScore.swift`            | Silhouette coefficient for cluster cohesion & separation |
| **Clustering Metrics**   | `SwiftCluster/Metrics/ClusteringMetrics.swift`          | Inertia, Calinski-Harabasz, Davies-Bouldin, ARI, NMI     |
| **Supervised Metrics**   | `SwiftOptimize/Metrics.swift`                           | ROC-AUC, PR-AUC, MCC, Log-Loss, R², Adjusted R², MAPE  |
| **Validation Folds**     | `SwiftOptimize/Validation/StratifiedKFold.swift`        | Class-ratio preserving K-Fold split                      |
| **Validation Folds**     | `SwiftOptimize/Validation/TimeSeriesSplit.swift`        | Expanding-window temporal split                          |
| **Validation Folds**     | `SwiftOptimize/Validation/GroupKFold.swift`             | Non-overlapping group fold generator                     |
| **Feature Transformers** | `SwiftPreprocessing/Core/PolynomialFeatures.swift`      | Polynomial & interaction features                        |
| **Feature Selectors**    | `SwiftPreprocessing/Core/FeatureSelection.swift`        | VarianceThreshold, SelectKBest, RFE                      |
| **Gradient Boosting**    | `SwiftML/Ensembles/HistGradientBoostingRegressor.swift` | Histogram binned gradient boosting                       |
| **Calibration**          | `SwiftML/Calibration/ProbabilityCalibration.swift`      | Isotonic regression & Platt scaling                      |
| **Survival Analysis**    | `SwiftStats/Survival/SurvivalAnalysis.swift`            | Kaplan-Meier & Cox Proportional Hazards                  |
| **Exporters**            | `SwiftML/Serialization/CoreMLExporter.swift`            | CoreML`.mlmodel` exporter                              |
| **Exporters**            | `SwiftML/Serialization/ONNXExporter.swift`              | ONNX format exporter                                     |

---

## Verification Plan

### Automated Unit Tests

- `swift test --filter SilhouetteScoreTests`: Verify Silhouette coefficient on synthetic Gaussian blobs vs overlapping noise.
- `swift test --filter ClusteringMetricsTests`: Validate Inertia, Davies-Bouldin, and Calinski-Harabasz against scikit-learn outputs.
- `swift test --filter MetricsTests`: Verify exact numeric parity for ROC-AUC, PR-AUC, MCC, R², and MAPE.
- `swift test --filter ValidationTests`: Test class distribution ratio in `StratifiedKFold` and check expanding window bounds in `TimeSeriesSplit`.

### Manual & UI Verification

- Test Saura `ModelLeaderboardView.swift` to verify dynamic metric column rendering based on task type.
