#!/usr/bin/env python3
"""
SwiftAnalytics Python Benchmark Suite — v0.7
Mirrors the Swift benchmarks in Benchmarks/Swift/ for direct comparison.

Usage:
    pip install -r requirements.txt
    python3 benchmarks.py                        # console output only
    python3 benchmarks.py --json python_results.json

All datasets are generated with numpy.random.seed(42) so they are
deterministically equivalent to the LCG data in the Swift side.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose


# ── Measurement harness ────────────────────────────────────────────────────────

def measure(fn, warmup: int = 2, iterations: int = 7):
    """Returns (mean_ms, median_ms, min_ms, max_ms, std_ms)."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)   # → milliseconds

    arr = np.array(times)
    return {
        "meanMs":   float(np.mean(arr)),
        "medianMs": float(np.median(arr)),
        "minMs":    float(np.min(arr)),
        "maxMs":    float(np.max(arr)),
        "stdMs":    float(np.std(arr)),
        "iterations": iterations,
        "warmup": warmup,
    }


def run_benchmark(name: str, module: str, fn, warmup=2, iterations=7) -> dict:
    stats = measure(fn, warmup=warmup, iterations=iterations)
    result = {"name": name, "module": module, **stats}
    print(f"  ✓ {name:<52}  {stats['medianMs']:8.3f} ms (median)")
    return result


# ── SwiftStats equivalent: NumPy ──────────────────────────────────────────────

def bench_stats():
    print("▶ Running SwiftStats (NumPy) benchmarks …")
    np.random.seed(42)
    data = np.random.uniform(-50.0, 50.0, size=1_000_000)
    data_b = np.random.uniform(-50.0, 50.0, size=500_000)
    data_a = np.random.uniform(-50.0, 50.0, size=500_000)

    results = []
    results.append(run_benchmark(
        "Mean (NumPy, 1M elements)", "NumPy",
        lambda: np.mean(data)
    ))
    results.append(run_benchmark(
        "StdDev (NumPy, 1M elements)", "NumPy",
        lambda: np.std(data, ddof=1)
    ))
    results.append(run_benchmark(
        "Variance (NumPy, 1M elements)", "NumPy",
        lambda: np.var(data, ddof=1)
    ))
    results.append(run_benchmark(
        "Pearson Correlation (NumPy, 500k)", "NumPy",
        lambda: np.corrcoef(data_a, data_b)
    ))
    print()
    return results


# ── SwiftDataFrame equivalent: Pandas ─────────────────────────────────────────

def bench_dataframe():
    print("▶ Running SwiftDataFrame (Pandas) benchmarks …")
    np.random.seed(42)

    n = 100_000
    categories = np.random.choice(["alpha", "beta", "gamma", "delta"], size=n)
    df_full = pd.DataFrame({
        "id":       np.arange(n),
        "category": categories,
        "value_a":  np.random.uniform(0, 100, size=n),
        "value_b":  np.random.uniform(0, 50, size=n),
        "flag":     np.where(np.arange(n) % 2 == 0, True, False),
    })

    # Write CSV once for read benchmark
    csv_path = "/tmp/swiftanalytics_bench_python.csv"
    df_full.to_csv(csv_path, index=False)

    results = []
    results.append(run_benchmark(
        "CSV Read (100k rows, 5 cols)", "Pandas",
        lambda: pd.read_csv(csv_path),
        warmup=1, iterations=5
    ))
    results.append(run_benchmark(
        "Filter rows (predicate, 100k rows)", "Pandas",
        lambda: df_full[df_full["value_a"] > 50.0],
        iterations=7
    ))
    results.append(run_benchmark(
        "GroupBy + sum/mean (4 groups)", "Pandas",
        lambda: df_full.groupby("category").agg({"value_a": "sum", "value_b": "mean"}),
        iterations=7
    ))
    results.append(run_benchmark(
        "SortBy double column (100k rows)", "Pandas",
        lambda: df_full.sort_values("value_a"),
        iterations=7
    ))
    print()
    return results


# ── SwiftML equivalent: Scikit-Learn ──────────────────────────────────────────

def bench_ml():
    print("▶ Running SwiftML (Scikit-Learn) benchmarks …")
    np.random.seed(42)

    # Linear regression: 10k × 10
    X_lr = np.random.uniform(-5, 5, size=(10_000, 10))
    weights = np.arange(1, 11, dtype=float)
    y_lr = X_lr @ weights + 1.0

    # Classification: 1k × 4
    X_clf = np.random.uniform(-5, 5, size=(1_000, 4))
    y_clf = ((X_clf[:, 0] > 0) & (X_clf[:, 1] > 0)).astype(int)

    # Regression: 1k × 4
    X_reg = np.random.uniform(-5, 5, size=(1_000, 4))
    w4 = np.array([1.0, 2.0, 3.0, 4.0])
    y_reg = X_reg @ w4 + 1.0

    # Cluster: 10k × 4
    X_km = np.random.uniform(-10, 10, size=(10_000, 4))

    # PCA: 1k × 100
    X_pca = np.random.uniform(-10, 10, size=(1_000, 100))

    results = []
    # Match SwiftML LinearRegression: gradient descent for 100 epochs (not closed-form OLS).
    results.append(run_benchmark(
        "LinearRegression fit (10k×10, 100 epochs)", "Scikit-Learn",
        lambda: SGDRegressor(
            loss="squared_error",
            penalty=None,
            learning_rate="constant",
            eta0=0.01,
            max_iter=100,
            tol=None,
            fit_intercept=True,
            shuffle=True,
            random_state=42,
        ).fit(X_lr, y_lr),
        warmup=1, iterations=5
    ))
    results.append(run_benchmark(
        "RandomForest fit (1k×4, 50 trees)", "Scikit-Learn",
        lambda: RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42).fit(X_clf, y_clf),
        warmup=1, iterations=5
    ))
    results.append(run_benchmark(
        "GBDT Regressor fit (1k×4, 50 est.)", "Scikit-Learn",
        lambda: GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42).fit(X_reg, y_reg),
        warmup=1, iterations=5
    ))
    results.append(run_benchmark(
        "KMeans fit (10k×4, 3 clusters)", "Scikit-Learn",
        lambda: KMeans(n_clusters=3, max_iter=50, n_init=1, random_state=42).fit(X_km),
        warmup=1, iterations=5
    ))
    results.append(run_benchmark(
        "PCA SVD fit (1k×100 → 10 comps)", "Scikit-Learn",
        lambda: PCA(n_components=10).fit_transform(X_pca),
        warmup=1, iterations=5
    ))
    print()
    return results


# ── SwiftForecast equivalent: Statsmodels ─────────────────────────────────────

def bench_forecast():
    print("▶ Running SwiftForecast (Statsmodels) benchmarks …")
    np.random.seed(42)

    # Seasonal series
    t = np.arange(1_000)
    hw_series = 20.0 + 0.3 * t + 5.0 * np.sin(2 * np.pi * t / 12) + np.random.uniform(-0.5, 0.5, 1_000)

    # Random walk
    steps = np.random.uniform(-1, 1, 500)
    arima_series = np.cumsum(np.insert(steps, 0, 0.0))[:500]

    # Constant-velocity 1D Kalman (matches SwiftForecast.KalmanFilter.oneDimensional)
    obs = np.random.uniform(24.0, 26.0, 10_000)

    def kalman_1d_filter(observations: np.ndarray,
                         process_noise: float = 0.05,
                         measurement_noise: float = 1.0) -> np.ndarray:
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * process_noise
        R = np.array([[measurement_noise]])
        x = np.array([25.0, 0.0])
        P = np.eye(2)
        states = np.empty((len(observations), 2))
        eye = np.eye(2)
        for i, z in enumerate(observations):
            x = F @ x
            P = F @ P @ F.T + Q
            y = np.array([z]) - (H @ x)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + (K @ y).ravel()
            P = (eye - K @ H) @ P
            states[i] = x
        return states

    results = []
    results.append(run_benchmark(
        "Holt-Winters fit (1k pts, period=12)", "Statsmodels",
        lambda: ExponentialSmoothing(hw_series, trend="add", seasonal="add", seasonal_periods=12).fit(optimized=True),
        warmup=1, iterations=5
    ))
    results.append(run_benchmark(
        "ARIMA(1,1,1) fit (500 pts)", "Statsmodels",
        lambda: ARIMA(arima_series, order=(1, 1, 1)).fit(),
        warmup=1, iterations=5
    ))

    def arima_fit_and_forecast():
        m = ARIMA(arima_series, order=(1, 1, 1)).fit()
        m.forecast(steps=24)

    results.append(run_benchmark(
        "ARIMA(1,1,1) forecast horizon=24", "Statsmodels",
        arima_fit_and_forecast,
        warmup=1, iterations=5
    ))
    results.append(run_benchmark(
        "Kalman Filter 1D (10k obs)", "NumPy",
        lambda: kalman_1d_filter(obs),
        warmup=1, iterations=5
    ))
    results.append(run_benchmark(
        "TS Decomposition additive (1k pts)", "Statsmodels",
        lambda: seasonal_decompose(hw_series, model="additive", period=12),
        warmup=1, iterations=7
    ))
    print()
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SwiftAnalytics Python Benchmark Suite")
    parser.add_argument("--json", metavar="PATH", help="Export results to JSON file")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║      SwiftAnalytics Python Benchmark Suite — v0.7           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"Platform : {platform.machine()} ({platform.system()})")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"NumPy    : {np.__version__}")
    print(f"Pandas   : {pd.__version__}")
    print("")

    all_results = []
    all_results.extend(bench_stats())
    all_results.extend(bench_dataframe())
    all_results.extend(bench_ml())
    all_results.extend(bench_forecast())

    # Console summary table
    print(f"\n{'Benchmark':<52}  {'Module':<14}  {'Mean(ms)':>10}  {'Median(ms)':>10}")
    print("─" * 96)
    for r in all_results:
        print(f"  {r['name']:<50}  {r['module']:<14}  {r['meanMs']:10.3f}  {r['medianMs']:10.3f}")
    print("─" * 96)

    if args.json:
        report = {
            "platform": f"{platform.machine()} ({platform.system()})",
            "pythonVersion": sys.version.split()[0],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": all_results,
        }
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Results exported to: {args.json}\n")


if __name__ == "__main__":
    main()
