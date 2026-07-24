# Missing Evaluation Metrics in SwiftSci & Saura

This document outlines all evaluation metrics that are currently missing or unhandled in **SwiftSci** and **Saura**, categorized by machine learning task type.

---

## 1. Unsupervised Learning & Clustering Metrics (`SwiftCluster`)

Currently, unsupervised models (`K-Means`, `Isolation Forest`, `DBSCAN`) display `0.0000` for `Accuracy`, `F1 Score`, `Precision`, and `Recall` because supervised metrics are not applicable without ground-truth labels.

### Missing Metrics:
- **Silhouette Coefficient / Score**: Measures cluster cohesion and separation (\([-1, 1]\)).
- **Inertia / WCSS (Within-Cluster Sum of Squares)**: Sum of squared distances of samples to their closest cluster center (for K-Means).
- **Calinski-Harabasz Index (Variance Ratio Criterion)**: Ratio of between-clusters dispersion to within-cluster dispersion.
- **Davies-Bouldin Index**: Average similarity measure of each cluster with its most similar cluster (lower values indicate better clustering).
- **Contamination Ratio / Anomaly Score**: Percentage of dataset samples classified as outliers in `Isolation Forest` based on average path length \(s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}\).
- **Adjusted Rand Index (ARI)** & **Normalized Mutual Information (NMI)**: External validation metrics when true ground-truth cluster labels are provided.

---

## 2. Classification Metrics (`SwiftML` & `SwiftStats`)

### Missing Metrics:
- **ROC-AUC (Receiver Operating Characteristic Area Under Curve)**: Evaluates binary classification performance across all classification thresholds.
- **PR-AUC / Average Precision**: Area under the Precision-Recall curve, essential for highly imbalanced classification datasets.
- **Matthews Correlation Coefficient (MCC)**: Balanced measure taking into account true and false positives and negatives, robust against class imbalance (\([-1, 1]\)).
- **Log-Loss / Cross-Entropy Loss**: Logarithmic loss metric measuring probability confidence.
- **Balanced Accuracy**: Macro-averaged recall across classes to prevent majority class bias.
- **F-beta Score (\(F_{0.5}\), \(F_2\))**: Weighted harmonic mean prioritizing precision (\(\beta=0.5\)) or recall (\(\beta=2\)).

---

## 3. Regression Metrics (`SwiftStats` & `SwiftML`)

### Missing Metrics:
- **R² Score (Coefficient of Determination)**: Proportion of variance in target variable predictable from features.
- **Adjusted R²**: Penalized \(R^2\) score taking feature count into account.
- **MAPE (Mean Absolute Percentage Error)**: Relative error metric expressed as a percentage.
- **Explained Variance Score**: Measures proportion of target variation explained by model predictions.

---

## 4. UI Adaptation Roadmap (Saura Model Leaderboard)

- Dynamic metric selector in `Model Leaderboard` depending on model task:
  - **Classification**: `Accuracy`, `F1 Score`, `Precision`, `Recall`, `ROC-AUC`.
  - **Regression**: `R²`, `RMSE`, `MAE`, `MSE`.
  - **Unsupervised / Clustering**: `Silhouette Score`, `Inertia`, `Contamination Ratio`.
