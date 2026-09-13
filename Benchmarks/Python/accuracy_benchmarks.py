#!/usr/bin/env python3
"""
SwiftSci Python Accuracy Benchmarks — v3.8.1
===========================================
Direct counterpart to SwiftSci AccuracyBenchmarks.swift.

Uses the SAME BenchmarkLCG generator as the Swift side:
    state = (state * 6364136223846793005 + 1442695040888963407) mod 2^64
    double = (state >> 11) / (1 << 53)

This guarantees bit-identical datasets for a 100% fair apples-to-apples comparison.

Usage:
    python3 accuracy_benchmarks.py
    python3 accuracy_benchmarks.py --json accuracy_results.json
"""

import argparse
import json
import math
import sys
import numpy as np
from sklearn.linear_model import LinearRegression as SkLinearRegression, LogisticRegression as SkLogisticRegression
from sklearn.svm import LinearSVC as SkLinearSVC
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA as SkPCA
from sklearn.cluster import KMeans as SkKMeans
from sklearn.preprocessing import StandardScaler as SkStandardScaler, MinMaxScaler as SkMinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score
)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    HAS_VADER = True
except ImportError:
    HAS_VADER = False


# ── Replicated BenchmarkLCG from Swift ────────────────────────────────────────

class BenchmarkLCG:
    """
    Exact port of BenchmarkLCG from BenchmarkSuite.swift.
    state = (state * A + C) mod 2^64
    double = (state >> 11) / (1 << 53)  → [0, 1)
    """
    A = 6_364_136_223_846_793_005
    C = 1_442_695_040_888_963_407
    MOD = 1 << 64

    def __init__(self, seed: int = 42):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self.state = (self.state * self.A + self.C) % self.MOD
        return self.state

    def next_double(self, lo: float = 0.0, hi: float = 1.0) -> float:
        normalized = (self.next() >> 11) / (1 << 53)
        return lo + normalized * (hi - lo)


# ── Main benchmark runner ─────────────────────────────────────────────────────

def run_accuracy_benchmarks():
    rng = BenchmarkLCG(seed=42)

    line_width = 82
    border = "─" * (line_width + 2)

    print("  ┌" + border + "┐")
    title = "PYTHON BASELINE ACCURACY & QUALITY SCORECARD"
    pad_left = (line_width - len(title)) // 2
    pad_right = line_width - len(title) - pad_left
    print(f"  │ {' ' * pad_left}{title}{' ' * pad_right} │")

    def _section(sec_title):
        print("  ├" + border + "┤")
        t = "▶ " + sec_title
        print(f"  │ {t}{' ' * max(0, line_width - len(t))} │")
        print("  ├" + border + "┤")

    def _row(text):
        t = text[:line_width] if len(text) > line_width else text
        print(f"  │ {t}{' ' * max(0, line_width - len(t))} │")

    results = {}

    # ══════════════════════════════════════════════════════════════════
    # ── 1. Time-Series Forecasting (Holt-Winters & ARIMA) ─────────────
    # ══════════════════════════════════════════════════════════════════
    _section("1. TIME-SERIES FORECASTING (Horizon = 24)")

    series = []
    for t in range(500):
        trend = t * 0.25
        seasonal = 8.0 * math.sin(t * 2.0 * math.pi / 12.0)
        noise = rng.next_double(-0.5, 0.5)
        series.append(20.0 + trend + seasonal + noise)

    train_series = series[:476]
    actual_future = np.array(series[476:])

    # Holt-Winters
    hw_model = ExponentialSmoothing(
        train_series, trend="add", seasonal="add", seasonal_periods=12
    ).fit()
    hw_pred = hw_model.forecast(24)

    hw_rmse = math.sqrt(mean_squared_error(actual_future, hw_pred))
    hw_mae  = mean_absolute_error(actual_future, hw_pred)
    hw_mape = float(np.mean(np.abs((actual_future - hw_pred) / actual_future))) * 100.0
    hw_r2   = r2_score(actual_future, hw_pred)

    _row(f"[Forecast] Holt-Winters (h=24) : RMSE={hw_rmse:.3f}, MAE={hw_mae:.3f}, MAPE={hw_mape:.2f}%, R²={hw_r2:.3f}")
    results["holt_winters"] = {"rmse": hw_rmse, "mae": hw_mae, "mape": hw_mape, "r2": hw_r2}

    # ARIMA(1,1,1)
    arima_model = ARIMA(train_series, order=(1, 1, 1)).fit()
    arima_pred  = arima_model.forecast(24)

    arima_rmse = math.sqrt(mean_squared_error(actual_future, arima_pred))
    arima_mae  = mean_absolute_error(actual_future, arima_pred)
    arima_mape = float(np.mean(np.abs((actual_future - arima_pred) / actual_future))) * 100.0
    arima_r2   = r2_score(actual_future, arima_pred)

    _row(f"[Forecast] ARIMA(1,1,1) (h=24) : RMSE={arima_rmse:.3f}, MAE={arima_mae:.3f}, MAPE={arima_mape:.2f}%, R²={arima_r2:.3f}")
    results["arima"] = {"rmse": arima_rmse, "mae": arima_mae, "mape": arima_mape, "r2": arima_r2}

    # ══════════════════════════════════════════════════════════════════
    # ── 2. Supervised Regression (OLS, GBDT, HistGBDT, DT) ────────────
    # ══════════════════════════════════════════════════════════════════
    _section("2. SUPERVISED REGRESSION (N = 1000, 80/20 split)")

    # 2.1 Linear OLS
    lin_X, lin_Y = [], []
    for _ in range(1000):
        x1 = rng.next_double(-2.0, 2.0)
        x2 = rng.next_double(-2.0, 2.0)
        x3 = rng.next_double(-2.0, 2.0)
        y = 3.0 * x1 - 2.0 * x2 + 1.5 * x3 + 0.5 + rng.next_double(-0.1, 0.1)
        lin_X.append([x1, x2, x3])
        lin_Y.append(y)
    lin_X, lin_Y = np.array(lin_X), np.array(lin_Y)
    lin_X_tr, lin_X_te = lin_X[:800], lin_X[800:]
    lin_Y_tr, lin_Y_te = lin_Y[:800], lin_Y[800:]

    ols = SkLinearRegression()
    ols.fit(lin_X_tr, lin_Y_tr)
    ols_pred = ols.predict(lin_X_te)
    ols_rmse = math.sqrt(mean_squared_error(lin_Y_te, ols_pred))
    ols_mae  = mean_absolute_error(lin_Y_te, ols_pred)
    ols_r2   = r2_score(lin_Y_te, ols_pred)

    _row(f"[ML Reg]   OLS Linear (LAPACK)  : RMSE={ols_rmse:.4f}, MAE={ols_mae:.4f}, R²={ols_r2:.5f}")
    results["linear_ols"] = {"rmse": ols_rmse, "mae": ols_mae, "r2": ols_r2, "coef": ols.coef_.tolist(), "intercept": float(ols.intercept_)}

    # 2.2 Non-linear regression data
    all_X, all_Y = [], []
    for _ in range(1000):
        x1 = rng.next_double(-3.0, 3.0)
        x2 = rng.next_double(-3.0, 3.0)
        y = 2.0 * x1 + math.sin(x2) * 3.0 + rng.next_double(-0.2, 0.2)
        all_X.append([x1, x2])
        all_Y.append(y)
    all_X, all_Y = np.array(all_X), np.array(all_Y)
    x_tr, x_te = all_X[:800], all_X[800:]
    y_tr, y_te = all_Y[:800], all_Y[800:]

    # GBDT
    gbdt = GradientBoostingRegressor(n_estimators=30, max_depth=4, learning_rate=0.1, random_state=0)
    gbdt.fit(x_tr, y_tr)
    gbdt_pred = gbdt.predict(x_te)
    gbdt_rmse = math.sqrt(mean_squared_error(y_te, gbdt_pred))
    gbdt_r2   = r2_score(y_te, gbdt_pred)

    _row(f"[ML Reg]   GBDT (30 tr, d=4)    : RMSE={gbdt_rmse:.3f}, R²={gbdt_r2:.4f}")
    results["gbdt"] = {"rmse": gbdt_rmse, "r2": gbdt_r2}

    # HistGradientBoosting
    hgbr = HistGradientBoostingRegressor(max_iter=30, max_depth=4, learning_rate=0.1, max_bins=255, random_state=0)
    hgbr.fit(x_tr, y_tr)
    hgbr_pred = hgbr.predict(x_te)
    hgbr_rmse = math.sqrt(mean_squared_error(y_te, hgbr_pred))
    hgbr_r2   = r2_score(y_te, hgbr_pred)

    _row(f"[ML Reg]   HistGBDT (30 tr, b255): RMSE={hgbr_rmse:.3f}, R²={hgbr_r2:.4f}")
    results["hist_gbdt_reg"] = {"rmse": hgbr_rmse, "r2": hgbr_r2}

    # DecisionTree
    dtr = DecisionTreeRegressor(max_depth=5, min_samples_split=2, random_state=0)
    dtr.fit(x_tr, y_tr)
    dtr_pred = dtr.predict(x_te)
    dtr_rmse = math.sqrt(mean_squared_error(y_te, dtr_pred))
    dtr_r2   = r2_score(y_te, dtr_pred)

    _row(f"[ML Reg]   DecisionTree (d=5)   : RMSE={dtr_rmse:.3f}, R²={dtr_r2:.4f}")
    results["decision_tree_reg"] = {"rmse": dtr_rmse, "r2": dtr_r2}

    # ══════════════════════════════════════════════════════════════════
    # ── 3. Supervised Classification (LogReg, SVC, RF, HistGB, DT, NB)
    # ══════════════════════════════════════════════════════════════════
    _section("3. SUPERVISED CLASSIFICATION (N = 1000, 80/20 split)")

    cls_X, cls_Y = [], []
    for _ in range(1000):
        x1 = rng.next_double(-2.0, 2.0)
        x2 = rng.next_double(-2.0, 2.0)
        label = 1 if (x1 * 0.8 + x2 * 0.6 > 0.0) else 0
        cls_X.append([x1, x2])
        cls_Y.append(label)
    cls_X, cls_Y = np.array(cls_X), np.array(cls_Y)
    cls_X_tr, cls_X_te = cls_X[:800], cls_X[800:]
    cls_Y_tr, cls_Y_te = cls_Y[:800], cls_Y[800:]

    # Logistic Regression
    logreg = SkLogisticRegression(max_iter=500, random_state=0)
    logreg.fit(cls_X_tr, cls_Y_tr)
    logreg_pred = logreg.predict(cls_X_te)
    logreg_acc = accuracy_score(cls_Y_te, logreg_pred)
    logreg_f1  = f1_score(cls_Y_te, logreg_pred, pos_label=1)

    _row(f"[ML Cls]   LogisticRegression   : Accuracy={logreg_acc * 100.0:.2f}%, F1={logreg_f1:.3f}")
    results["logistic_regression"] = {"accuracy": logreg_acc, "f1": logreg_f1}

    # LinearSVC
    svc = SkLinearSVC(C=1.0, dual="auto", max_iter=1000, random_state=0)
    svc.fit(cls_X_tr, cls_Y_tr)
    svc_pred = svc.predict(cls_X_te)
    svc_acc = accuracy_score(cls_Y_te, svc_pred)
    svc_f1  = f1_score(cls_Y_te, svc_pred, pos_label=1)

    _row(f"[ML Cls]   LinearSVC (C=1.0)    : Accuracy={svc_acc * 100.0:.2f}%, F1={svc_f1:.3f}")
    results["linear_svc"] = {"accuracy": svc_acc, "f1": svc_f1}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=30, max_depth=5, criterion="gini", random_state=0)
    rf.fit(cls_X_tr, cls_Y_tr)
    rf_pred = rf.predict(cls_X_te)
    rf_acc = accuracy_score(cls_Y_te, rf_pred)
    rf_f1  = f1_score(cls_Y_te, rf_pred, pos_label=1)

    _row(f"[ML Cls]   RandomForest (30 tr.): Accuracy={rf_acc * 100.0:.2f}%, F1={rf_f1:.3f}")
    results["random_forest"] = {"accuracy": rf_acc, "f1": rf_f1}

    # HistGradientBoosting Classifier
    hgbc = HistGradientBoostingClassifier(max_iter=30, max_depth=4, learning_rate=0.1, max_bins=255, random_state=0)
    hgbc.fit(cls_X_tr, cls_Y_tr)
    hgbc_pred = hgbc.predict(cls_X_te)
    hgbc_acc = accuracy_score(cls_Y_te, hgbc_pred)
    hgbc_f1  = f1_score(cls_Y_te, hgbc_pred, pos_label=1)

    _row(f"[ML Cls]   HistGBDT Classifier  : Accuracy={hgbc_acc * 100.0:.2f}%, F1={hgbc_f1:.3f}")
    results["hist_gbdt_cls"] = {"accuracy": hgbc_acc, "f1": hgbc_f1}

    # DecisionTree Classifier
    dtc = DecisionTreeClassifier(max_depth=5, min_samples_split=2, criterion="gini", random_state=0)
    dtc.fit(cls_X_tr, cls_Y_tr)
    dtc_pred = dtc.predict(cls_X_te)
    dtc_acc = accuracy_score(cls_Y_te, dtc_pred)
    dtc_f1  = f1_score(cls_Y_te, dtc_pred, pos_label=1)

    _row(f"[ML Cls]   DecisionTree (d=5)   : Accuracy={dtc_acc * 100.0:.2f}%, F1={dtc_f1:.3f}")
    results["decision_tree_cls"] = {"accuracy": dtc_acc, "f1": dtc_f1}

    # Naive Bayes
    x_nb_tr = np.array([[(i + j) % 5 for j in range(50)] for i in range(800)], dtype=float)
    y_nb_tr = np.array([i % 3 for i in range(800)])
    x_nb_te = np.array([[(i + j) % 5 for j in range(50)] for i in range(200)], dtype=float)
    y_nb_te = np.array([i % 3 for i in range(200)])

    nb = MultinomialNB(alpha=1.0)
    nb.fit(x_nb_tr, y_nb_tr)
    nb_pred = nb.predict(x_nb_te)
    nb_acc = accuracy_score(y_nb_te, nb_pred)
    nb_f1  = f1_score(y_nb_te, nb_pred, average="macro")

    _row(f"[NLP Cls]  NaiveBayes (3-class) : Accuracy={nb_acc * 100.0:.2f}%, Macro-F1={nb_f1:.3f}")
    results["naive_bayes"] = {"accuracy": nb_acc, "macro_f1": nb_f1}

    # ══════════════════════════════════════════════════════════════════
    # ── 4. Unsupervised Clustering & SVD (PCA & K-Means) ──────────────
    # ══════════════════════════════════════════════════════════════════
    _section("4. UNSUPERVISED LEARNING & SPECTRAL DECOMPOSITION")

    # PCA
    pca_data = []
    for _ in range(500):
        z1 = rng.next_double(-3.0, 3.0)
        z2 = rng.next_double(-2.0, 2.0)
        x0 = z1 * 2.0 + rng.next_double(-0.1, 0.1)
        x1 = z1 * 1.5 + z2 * 0.5 + rng.next_double(-0.1, 0.1)
        x2 = z2 * 3.0 + rng.next_double(-0.1, 0.1)
        x3 = -z1 + z2 * 0.8 + rng.next_double(-0.1, 0.1)
        x4 = rng.next_double(-0.5, 0.5)
        pca_data.append([x0, x1, x2, x3, x4])
    pca_data = np.array(pca_data)

    pca = SkPCA(n_components=2, svd_solver="full")
    pca.fit(pca_data)
    evr = pca.explained_variance_ratio_
    tot_evr = float(np.sum(evr))

    _row(f"[Cluster]  PCA SVD (5D → 2D)    : EVR=[{evr[0]:.4f}, {evr[1]:.4f}], Total={tot_evr * 100.0:.2f}%")
    results["pca"] = {"evr": evr.tolist(), "singular_values": pca.singular_values_.tolist()}

    # K-Means
    km_data = []
    for i in range(600):
        cluster_idx = i % 3
        cx = float(cluster_idx) * 5.0
        cy = float(cluster_idx) * -3.0
        x = cx + rng.next_double(-1.0, 1.0)
        y = cy + rng.next_double(-1.0, 1.0)
        km_data.append([x, y])
    km_data = np.array(km_data)

    kmeans = SkKMeans(n_clusters=3, max_iter=100, random_state=42, n_init=1)
    kmeans.fit(km_data)
    inertia = kmeans.inertia_

    _row(f"[Cluster]  K-Means (k=3, N=600) : Inertia (WCSS)={inertia:.2f}, Centroids=3")
    results["kmeans"] = {"inertia": inertia}

    # ══════════════════════════════════════════════════════════════════
    # ── 5. Feature Preprocessing (StandardScaler & MinMaxScaler) ──────
    # ══════════════════════════════════════════════════════════════════
    _section("5. PREPROCESSING & IEEE 754 PRECISION (N = 1000)")

    scale_data = []
    for _ in range(1000):
        f1 = rng.next_double(10.0, 50.0)
        f2 = rng.next_double(-100.0, 100.0)
        f3 = rng.next_double(0.0, 1.0)
        scale_data.append([f1, f2, f3])
    scale_data = np.array(scale_data)

    # StandardScaler
    scaler = SkStandardScaler()
    scaled_data = scaler.fit_transform(scale_data)
    mean0 = scaler.mean_[0]
    std0  = scaler.scale_[0]
    post_scaled_mean = float(np.mean(scaled_data[:, 0]))

    _row(f"[Preproc]  StandardScaler       : col0 μ={mean0:.4f}, σ={std0:.4f} (Post-scaled μ={post_scaled_mean:.2e})")
    results["standard_scaler"] = {"mean0": mean0, "std0": std0, "post_mean": post_scaled_mean}

    # MinMaxScaler
    mm_scaler = SkMinMaxScaler(feature_range=(0.0, 1.0))
    mm_scaled = mm_scaler.fit_transform(scale_data)
    min0 = mm_scaler.data_min_[0]
    max0 = mm_scaler.data_max_[0]
    t_min0 = float(np.min(mm_scaled[:, 0]))
    t_max0 = float(np.max(mm_scaled[:, 0]))

    _row(f"[Preproc]  MinMaxScaler         : col0 [{min0:.2f}, {max0:.2f}] → Transformed [{t_min0:.4f}, {t_max0:.4f}]")
    results["minmax_scaler"] = {"min0": min0, "max0": max0}

    # ══════════════════════════════════════════════════════════════════
    # ── 6. Inferential Statistics & Hypothesis Tests ──────────────────
    # ══════════════════════════════════════════════════════════════════
    _section("6. INFERENTIAL STATISTICS & HYPOTHESIS TESTING")

    samp1 = [rng.next_double(4.5, 5.5) for _ in range(1000)]
    samp2 = [rng.next_double(4.3, 5.7) for _ in range(1000)]
    samp3 = [rng.next_double(4.8, 5.8) for _ in range(1000)]

    # Welch's t-test
    w_t, w_p = stats.ttest_ind(samp1, samp2, equal_var=False)
    _row(f"[Stats]    Welch's t-test       : t={w_t:.4f}, p={w_p:.6e}")
    results["welch_ttest"] = {"t": w_t, "p": w_p}

    # Student's t-test
    s_t, s_p = stats.ttest_ind(samp1, samp2, equal_var=True)
    _row(f"[Stats]    Student's t-test     : t={s_t:.4f}, p={s_p:.6e}")
    results["student_ttest"] = {"t": s_t, "p": s_p}

    # Paired t-test
    p_t, p_p = stats.ttest_rel(samp2, samp1)
    _row(f"[Stats]    Paired t-test        : t={p_t:.4f}, p={p_p:.6e}")
    results["paired_ttest"] = {"t": p_t, "p": p_p}

    # ANOVA
    a_f, a_p = stats.f_oneway(samp1, samp2, samp3)
    _row(f"[Stats]    One-Way ANOVA        : F={a_f:.4f}, p={a_p:.6e}")
    results["anova"] = {"f": a_f, "p": a_p}

    # Correlation
    r_val, _ = stats.pearsonr(samp1, samp2)
    rho_val, _ = stats.spearmanr(samp1, samp2)
    _row(f"[Stats]    Correlation          : Pearson r={r_val:.5f}, Spearman ρ={rho_val:.5f}")
    results["correlation"] = {"pearson": r_val, "spearman": rho_val}

    # ══════════════════════════════════════════════════════════════════
    # ── 7. NLP: Sentiment Analysis & Text Processing ──────────────────
    # ══════════════════════════════════════════════════════════════════
    _section("7. NLP: VADER SENTIMENT POLARITY SCORING")

    if HAS_VADER:
        sia = SentimentIntensityAnalyzer()
        s_pos = sia.polarity_scores("SwiftSci 3.8.1 is incredibly fast, robust and accurate!")
        s_neg = sia.polarity_scores("The algorithm failed completely with disastrous and horrible errors.")
        s_neu = sia.polarity_scores("The dataset contains standard numerical observations and measurements.")

        _row(f"[NLP]      VADER (Positive)     : Compound={s_pos['compound']:.4f}, Pos={s_pos['pos']:.3f}, Neg={s_pos['neg']:.3f}")
        _row(f"[NLP]      VADER (Negative)     : Compound={s_neg['compound']:.4f}, Pos={s_neg['pos']:.3f}, Neg={s_neg['neg']:.3f}")
        _row(f"[NLP]      VADER (Neutral)      : Compound={s_neu['compound']:.4f}, Neu={s_neu['neu']:.3f}")
        results["vader"] = {"positive": s_pos, "negative": s_neg, "neutral": s_neu}

    print("  └" + border + "┘\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="SwiftSci Python Accuracy Benchmarks v3.8.1")
    parser.add_argument("--json", metavar="PATH", help="Export accuracy results to JSON")
    args = parser.parse_args()

    results = run_accuracy_benchmarks()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results exported to: {args.json}\n")


if __name__ == "__main__":
    main()
