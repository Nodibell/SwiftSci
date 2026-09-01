#!/usr/bin/env python3
"""
SwiftSci Python Accuracy Benchmarks — v3.5.0
Direct counterpart to SwiftSci AccuracyBenchmarks.swift.

Uses the SAME BenchmarkLCG generator as the Swift side:
    state = (state * 6364136223846793005 + 1442695040888963407) mod 2^64
    double = (state >> 11) / (1 << 53)

This guarantees bit-identical datasets for a fair apples-to-apples comparison.

Usage:
    python3 accuracy_benchmarks.py
    python3 accuracy_benchmarks.py --json accuracy_results.json
"""

import argparse
import json
import math
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
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


# ── Dataset generators using LCG ─────────────────────────────────────────────

def make_seasonal_series(rng: BenchmarkLCG, n: int = 500):
    """Mirrors AccuracyBenchmarks.swift lines 42-47."""
    series = []
    for t in range(n):
        trend = t * 0.25
        seasonal = 8.0 * math.sin(t * 2.0 * math.pi / 12.0)
        noise = rng.next_double(-0.5, 0.5)
        series.append(20.0 + trend + seasonal + noise)
    return series


def make_regression_data(rng: BenchmarkLCG, n: int = 1000):
    """Mirrors AccuracyBenchmarks.swift lines 118-124."""
    X, Y = [], []
    for _ in range(n):
        x1 = rng.next_double(-3.0, 3.0)
        x2 = rng.next_double(-3.0, 3.0)
        y = 2.0 * x1 + math.sin(x2) * 3.0 + rng.next_double(-0.2, 0.2)
        X.append([x1, x2])
        Y.append(y)
    return np.array(X), np.array(Y)


def make_classification_data(rng: BenchmarkLCG, n: int = 1000):
    """Mirrors AccuracyBenchmarks.swift lines 159-164."""
    X, Y = [], []
    for _ in range(n):
        x1 = rng.next_double(-2.0, 2.0)
        x2 = rng.next_double(-2.0, 2.0)
        label = 1 if (x1 * 0.8 + x2 * 0.6 > 0.0) else 0
        X.append([x1, x2])
        Y.append(label)
    return np.array(X), np.array(Y)


def make_nb_data():
    """Mirrors AccuracyBenchmarks.swift lines 197-200 (pure deterministic, no RNG)."""
    x_train = np.array([[(i + j) % 5 for j in range(50)] for i in range(800)], dtype=float)
    y_train = np.array([i % 3 for i in range(800)])
    x_test  = np.array([[(i + j) % 5 for j in range(50)] for i in range(200)], dtype=float)
    y_test  = np.array([i % 3 for i in range(200)])
    return x_train, y_train, x_test, y_test


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
    print("  ├" + border + "┤")

    results = {}

    # ── 1. Holt-Winters ───────────────────────────────────────────────────────
    series = make_seasonal_series(rng, n=500)
    train_series = series[:476]
    actual_future = np.array(series[476:])

    hw_model = ExponentialSmoothing(
        train_series, trend="add", seasonal="add", seasonal_periods=12
    ).fit(smoothing_level=0.2, smoothing_trend=0.1, smoothing_seasonal=0.1, optimized=False)
    hw_pred = hw_model.forecast(24)

    hw_rmse = math.sqrt(mean_squared_error(actual_future, hw_pred))
    hw_mae  = mean_absolute_error(actual_future, hw_pred)
    hw_mape = float(np.mean(np.abs((actual_future - hw_pred) / actual_future))) * 100.0
    hw_r2   = r2_score(actual_future, hw_pred)

    def _row(text):
        t = text[:line_width] if len(text) > line_width else text
        print(f"  │ {t}{' ' * max(0, line_width - len(t))} │")

    _row(f"[Forecast] Holt-Winters (h=24) : RMSE={hw_rmse:.3f}, MAE={hw_mae:.3f}, MAPE={hw_mape:.2f}%, R²={hw_r2:.3f}")
    results["holt_winters"] = {"rmse": hw_rmse, "mae": hw_mae, "mape": hw_mape, "r2": hw_r2}

    # ── 2. ARIMA(1,1,1) ───────────────────────────────────────────────────────
    arima_model = ARIMA(train_series, order=(1, 1, 1)).fit()
    arima_pred  = arima_model.forecast(24)

    arima_rmse = math.sqrt(mean_squared_error(actual_future, arima_pred))
    arima_mae  = mean_absolute_error(actual_future, arima_pred)
    arima_mape = float(np.mean(np.abs((actual_future - arima_pred) / actual_future))) * 100.0
    arima_r2   = r2_score(actual_future, arima_pred)

    _row(f"[Forecast] ARIMA(1,1,1) (h=24) : RMSE={arima_rmse:.3f}, MAE={arima_mae:.3f}, MAPE={arima_mape:.2f}%, R²={arima_r2:.3f}")
    results["arima"] = {"rmse": arima_rmse, "mae": arima_mae, "mape": arima_mape, "r2": arima_r2}

    # ── 3. GBDT Regressor ─────────────────────────────────────────────────────
    x_reg, y_reg = make_regression_data(rng, n=1000)
    x_train_r, x_test_r = x_reg[:800], x_reg[800:]
    y_train_r, y_test_r = y_reg[:800], y_reg[800:]

    gbdt = GradientBoostingRegressor(n_estimators=30, max_depth=4, learning_rate=0.1, random_state=0)
    gbdt.fit(x_train_r, y_train_r)
    gbdt_pred = gbdt.predict(x_test_r)

    gbdt_rmse = math.sqrt(mean_squared_error(y_test_r, gbdt_pred))
    gbdt_mae  = mean_absolute_error(y_test_r, gbdt_pred)
    gbdt_r2   = r2_score(y_test_r, gbdt_pred)

    _row(f"[ML Reg]   GBDT (30 trees, d=4) : RMSE={gbdt_rmse:.3f}, MAE={gbdt_mae:.3f}, R²={gbdt_r2:.4f}")
    results["gbdt"] = {"rmse": gbdt_rmse, "mae": gbdt_mae, "r2": gbdt_r2}

    # ── 4. Random Forest ──────────────────────────────────────────────────────
    x_cls, y_cls = make_classification_data(rng, n=1000)
    x_train_c, x_test_c = x_cls[:800], x_cls[800:]
    y_train_c, y_test_c = y_cls[:800], y_cls[800:]

    rf = RandomForestClassifier(n_estimators=30, max_depth=5, criterion="gini", random_state=0)
    rf.fit(x_train_c, y_train_c)
    rf_pred = rf.predict(x_test_c)

    rf_acc = accuracy_score(y_test_c, rf_pred)
    rf_f1  = f1_score(y_test_c, rf_pred, pos_label=1, average="binary")

    _row(f"[ML Cls]   RandomForest (30 tr.): Accuracy={rf_acc * 100.0:.2f}%, F1={rf_f1:.3f}")
    results["random_forest"] = {"accuracy": rf_acc, "f1": rf_f1}

    # ── 5. Naive Bayes ────────────────────────────────────────────────────────
    x_nb_tr, y_nb_tr, x_nb_te, y_nb_te = make_nb_data()

    nb = MultinomialNB(alpha=1.0)
    nb.fit(x_nb_tr, y_nb_tr)
    nb_pred = nb.predict(x_nb_te)

    nb_acc = accuracy_score(y_nb_te, nb_pred)
    nb_f1  = f1_score(y_nb_te, nb_pred, average="macro")

    _row(f"[NLP Cls]  NaiveBayes (3-class) : Accuracy={nb_acc * 100.0:.2f}%, Macro-F1={nb_f1:.3f}")
    results["naive_bayes"] = {"accuracy": nb_acc, "macro_f1": nb_f1}

    # ── 6. VADER ──────────────────────────────────────────────────────────────
    if HAS_VADER:
        sia    = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores("SwiftSci 3.5.0 is incredibly fast and robust!")
        _row(f"[NLP Snt]  VADER               : Compound={scores['compound']:.4f}, Pos={scores['pos']:.3f}")
        results["vader"] = scores

    # ── 7. Welch's t-test ─────────────────────────────────────────────────────
    samp1 = [rng.next_double(4.5, 5.5) for _ in range(1000)]
    samp2 = [rng.next_double(4.3, 5.7) for _ in range(1000)]
    t_stat, t_pval = stats.ttest_ind(samp1, samp2, equal_var=False)
    _row(f"[Stats]    Welch's t-test       : t={t_stat:.4f}, p={t_pval:.6e}")
    results["welch_ttest"] = {"t_stat": t_stat, "p_value": t_pval}

    print("  └" + border + "┘\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="SwiftSci Python Accuracy Benchmarks v3.5.0")
    parser.add_argument("--json", metavar="PATH", help="Export accuracy results to JSON")
    args = parser.parse_args()

    results = run_accuracy_benchmarks()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results exported to: {args.json}\n")


if __name__ == "__main__":
    main()
