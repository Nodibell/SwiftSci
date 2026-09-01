#!/usr/bin/env python3
"""
SwiftSci Python Benchmark Suite — v3.5.0
Mirrors the Swift benchmarks in Benchmarks/Swift/ for direct comparison.

Usage:
    python3 benchmarks.py                        # console output only
    python3 benchmarks.py --json python_results.json
    python3 benchmarks.py --suite ML
    python3 benchmarks.py --filter "Join"
    python3 benchmarks.py --rounds 3 --iterations 7

All datasets are generated with numpy.random.seed(42) so they are
deterministically equivalent to the LCG data on the Swift side.
"""

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import torch
import torch.nn as nn
import shap
from scipy import stats


# ── Configuration & Measurement harness ────────────────────────────────────────

class BenchmarkConfig:
    default_rounds = 3
    default_iterations = 7
    default_warmup = 2


def current_rss_mb() -> float:
    """Returns Resident Memory (RAM) in MB on macOS / Linux."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / (1024.0 * 1024.0)  # macOS returns bytes
    else:
        return usage / 1024.0             # Linux returns kilobytes


def measure(fn, warmup: int = 2, iterations: int = 7, rounds: int = 3):
    """
    Executes warmup iterations, then R rounds of I iterations.
    Computes Mean, 95% Confidence Interval, Trimmed Mean (20%), Median, Min..Max, and RAM.
    """
    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(rounds):
        for _ in range(iterations):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1000.0)   # → milliseconds

    arr = np.array(samples)
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std_val / math.sqrt(n) if n > 0 else 0.0
    margin_of_error_95 = 1.96 * se

    # 20% trimmed mean
    k = int(n * 0.2)
    trimmed_arr = np.sort(arr)[k:n-k] if n > 2 * k and k > 0 else arr
    trimmed_mean = float(np.mean(trimmed_arr))

    return {
        "meanMs":             mean_val,
        "marginOfError95Ms":  margin_of_error_95,
        "trimmedMeanMs":      trimmed_mean,
        "medianMs":           float(np.median(arr)),
        "minMs":              float(np.min(arr)),
        "maxMs":              float(np.max(arr)),
        "stdMs":              std_val,
        "rounds":             rounds,
        "iterations":         iterations,
        "warmup":             warmup,
        "memoryMB":           current_rss_mb(),
    }


def run_benchmark(name: str, module: str, fn, warmup=None, iterations=None, rounds=None) -> dict:
    w = warmup if warmup is not None else BenchmarkConfig.default_warmup
    it = iterations if iterations is not None else BenchmarkConfig.default_iterations
    r = rounds if rounds is not None else BenchmarkConfig.default_rounds

    print(f"  … {name:<52} [{r} rounds × {it} iters]", end="\r", flush=True)
    stats_dict = measure(fn, warmup=w, iterations=it, rounds=r)
    result = {"name": name, "module": module, **stats_dict}
    print(f"  ✓ {name:<52} {stats_dict['meanMs']:8.3f} ± {stats_dict['marginOfError95Ms']:5.3f} ms (mean, 95% CI) | {stats_dict['memoryMB']:5.1f} MB")
    return result


# ── SwiftStats equivalent: NumPy & SciPy ───────────────────────────────────────

def bench_stats():
    print("▶ Running SwiftStats (NumPy / SciPy) benchmarks …")
    np.random.seed(42)
    data = np.random.uniform(-50.0, 50.0, size=1_000_000)
    data_b = np.random.uniform(-50.0, 50.0, size=500_000)
    data_a = np.random.uniform(-50.0, 50.0, size=500_000)

    results = []
    results.append(run_benchmark("Mean (NumPy, 1M elements)", "NumPy", lambda: np.mean(data)))
    results.append(run_benchmark("StdDev (NumPy, 1M elements)", "NumPy", lambda: np.std(data, ddof=1)))
    results.append(run_benchmark("Variance (NumPy, 1M elements)", "NumPy", lambda: np.var(data, ddof=1)))
    results.append(run_benchmark("Pearson Correlation (NumPy, 500k)", "NumPy", lambda: np.corrcoef(data_a, data_b)))

    # SciPy T-Test & Spearman
    sample1 = np.random.uniform(-50.0, 50.0, size=100_000)
    sample2 = np.random.uniform(-50.0, 50.0, size=100_000)
    results.append(run_benchmark("Two-Sample T-Test (100k samples)", "SciPy", lambda: stats.ttest_ind(sample1, sample2, equal_var=False)))
    results.append(run_benchmark("Spearman Rank Correlation (100k pairs)", "SciPy", lambda: stats.spearmanr(sample1, sample2)))
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

    csv_path = "/tmp/swiftanalytics_bench_python.csv"
    df_full.to_csv(csv_path, index=False)

    results = []
    results.append(run_benchmark("CSV Read (100k rows, 5 cols)", "Pandas", lambda: pd.read_csv(csv_path), warmup=1, iterations=5))
    results.append(run_benchmark("CSV Stream Read (chunk=10k)", "Pandas", lambda: sum(len(c) for c in pd.read_csv(csv_path, chunksize=10000)), warmup=1, iterations=5))
    results.append(run_benchmark("CSV Stream + Filter", "Pandas", lambda: sum(len(c[c["value_a"] > 50.0]) for c in pd.read_csv(csv_path, chunksize=10000)), warmup=1, iterations=5))
    results.append(run_benchmark("CSV Stream + GroupBy", "Pandas", lambda: [c.groupby("category").agg({"value_a": "sum", "value_b": "mean"}) for c in pd.read_csv(csv_path, chunksize=10000)], warmup=1, iterations=5))
    results.append(run_benchmark("Filter rows (predicate, 100k rows)", "Pandas", lambda: df_full[df_full["value_a"] > 50.0], iterations=7))
    results.append(run_benchmark("GroupBy + sum/mean (4 groups)", "Pandas", lambda: df_full.groupby("category").agg({"value_a": "sum", "value_b": "mean"}), iterations=7))
    results.append(run_benchmark("SortBy double column (100k rows)", "Pandas", lambda: df_full.sort_values("value_a"), iterations=7))

    # Inner Join 100k rows
    df_right = pd.DataFrame({
        "id": np.arange(n),
        "weight": np.arange(n) * 0.05
    })
    results.append(run_benchmark("DataFrame SIMD Hash Join (100k rows)", "Pandas", lambda: pd.merge(df_full, df_right, on="id", how="inner"), iterations=5))

    # Parquet Write & Read 100k rows
    parquet_path = "/tmp/swiftanalytics_bench_python.parquet"
    results.append(run_benchmark("Parquet Write Snappy (100k rows)", "Pandas", lambda: df_full.to_parquet(parquet_path, engine="pyarrow", compression="snappy"), iterations=3))
    results.append(run_benchmark("Parquet Read Snappy (100k rows)", "Pandas", lambda: pd.read_parquet(parquet_path, engine="pyarrow"), iterations=5))
    print()
    return results


# ── SwiftML equivalent: Scikit-Learn ──────────────────────────────────────────

def bench_ml():
    print("▶ Running SwiftML (Scikit-Learn) benchmarks …")
    np.random.seed(42)

    # Linear regression
    n_lin, d_lin = 10_000, 10
    X_lin = np.random.uniform(-1.0, 1.0, size=(n_lin, d_lin))
    true_w = np.random.uniform(-1.0, 1.0, size=d_lin)
    y_lin = X_lin @ true_w + np.random.uniform(-0.1, 0.1, size=n_lin)

    # Random forest & GBDT
    n_rf, d_rf = 1_000, 4
    X_rf = np.random.uniform(-2.0, 2.0, size=(n_rf, d_rf))
    y_rf = (X_rf[:, 0] + X_rf[:, 1] > 0).astype(int)
    y_gbdt = X_rf[:, 0] * 2.0 + np.sin(X_rf[:, 1])

    # K-Means
    n_km, d_km = 10_000, 4
    X_km = np.random.uniform(-5.0, 5.0, size=(n_km, d_km))

    # PCA
    n_pca, d_pca = 1_000, 100
    X_pca = np.random.uniform(-1.0, 1.0, size=(n_pca, d_pca))

    results = []
    results.append(run_benchmark(
        "LinearRegression fit (10k×10, 100 epochs)", "Scikit-Learn",
        lambda: SGDRegressor(max_iter=100, tol=1e-6, random_state=42).fit(X_lin, y_lin),
        warmup=1, iterations=5
    ))
    results.append(run_benchmark(
        "RandomForest fit (1k×4, 50 trees)", "Scikit-Learn",
        lambda: RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1).fit(X_rf, y_rf),
        warmup=1, iterations=5
    ))
    results.append(run_benchmark(
        "GBDT Regressor fit (1k×4, 50 est.)", "Scikit-Learn",
        lambda: GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42).fit(X_rf, y_gbdt),
        warmup=1, iterations=5
    ))
    results.append(run_benchmark(
        "KMeans fit (10k×4, 3 clusters)", "Scikit-Learn",
        lambda: KMeans(n_clusters=3, max_iter=20, n_init=1, random_state=42).fit(X_km),
        warmup=1, iterations=5
    ))
    results.append(run_benchmark(
        "PCA SVD fit (1k×100 → 10 comps)", "Scikit-Learn",
        lambda: PCA(n_components=10, random_state=42).fit(X_pca),
        warmup=1, iterations=5
    ))

    # VectorStore Cosine Search (5k × 128d, top 10)
    embeddings = np.random.uniform(-1.0, 1.0, size=(5000, 128))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    query = np.random.uniform(-1.0, 1.0, size=(128,))
    query /= np.linalg.norm(query)
    def vector_search_fn():
        sims = np.dot(embeddings, query)
        _ = np.argpartition(sims, -10)[-10:]

    results.append(run_benchmark(
        "VectorStore Cosine Search (5k × 128d, top 10)", "NumPy",
        vector_search_fn,
        warmup=2, iterations=10
    ))
    print()
    return results


# ── SwiftForecast equivalent: Statsmodels ─────────────────────────────────────

def bench_forecast():
    print("▶ Running SwiftForecast (Statsmodels) benchmarks …")
    np.random.seed(42)

    # 50k points seasonal series
    t = np.arange(50_000, dtype=float)
    seasonal_50k = 20.0 + t * 0.3 + 5.0 * np.sin(t * 2.0 * np.pi / 12.0) + np.random.uniform(-0.5, 0.5, size=50_000)

    # 50k points random walk
    walk_50k = np.cumsum(np.random.uniform(-1.0, 1.0, size=50_000))

    results = []
    results.append(run_benchmark(
        "Holt-Winters fit (50k pts, period=12)", "Statsmodels",
        lambda: ExponentialSmoothing(seasonal_50k, seasonal_periods=12, trend="add", seasonal="add").fit(optimized=True),
        warmup=1, iterations=3
    ))
    results.append(run_benchmark(
        "ARIMA(1,1,1) fit (50k pts)", "Statsmodels",
        lambda: ARIMA(walk_50k, order=(1, 1, 1)).fit(),
        warmup=1, iterations=3
    ))
    results.append(run_benchmark(
        "TS Decomposition additive (1k pts)", "Statsmodels",
        lambda: seasonal_decompose(seasonal_50k[:1000], model="additive", period=12),
        warmup=1, iterations=7
    ))
    print()
    return results


# ── SwiftSci Extensions equivalent: Scikit-Learn, SHAP, NLTK ──────────────────

def bench_extensions():
    print("▶ Running SwiftSci Extensions (Scikit-Learn, SHAP, NLTK) benchmarks …")
    np.random.seed(42)

    # 1. Forecast & Regression Quality Error Metrics (100k)
    y_true = np.random.uniform(10.0, 100.0, size=100_000)
    y_pred = y_true + np.random.uniform(-5.0, 5.0, size=100_000)
    def errors_fn():
        _ = math.sqrt(mean_squared_error(y_true, y_pred))
        _ = mean_absolute_error(y_true, y_pred)
        _ = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
        _ = r2_score(y_true, y_pred)

    # 2. Classification ROC-AUC (50k)
    y_true_bin = (np.random.uniform(0, 1, size=50_000) > 0.5).astype(int)
    y_score = np.random.uniform(0.0, 1.0, size=50_000)
    def rocAuc_fn():
        _ = roc_auc_score(y_true_bin, y_score)

    # 3. OneHotEncoder (50k)
    cat_data = np.column_stack([
        [f"dept_{i % 8}" for i in range(50_000)],
        [f"region_{i % 4}" for i in range(50_000)]
    ])
    def ohe_fn():
        ohe = OneHotEncoder(sparse_output=False)
        _ = ohe.fit_transform(cat_data)

    # 4. NaiveBayes (1k × 100)
    nb_x = np.random.randint(0, 10, size=(1000, 100))
    nb_y = np.arange(1000) % 3
    def nb_fn():
        nb = MultinomialNB()
        nb.fit(nb_x, nb_y)

    # 5. KernelSHAP
    M = 5
    background = np.random.uniform(-2.0, 2.0, size=(20, M))
    instance = np.random.uniform(-2.0, 2.0, size=(1, M))
    explainer = shap.KernelExplainer(lambda x: np.sum(x, axis=1), background)
    def shap_fn():
        _ = explainer.shap_values(instance, nsamples=100, l1_reg=False)

    results = []
    results.append(run_benchmark("Forecast Errors Suite (RMSE, MAE, MAPE, R² 100k)", "Scikit-Learn", errors_fn, warmup=2, iterations=10))
    results.append(run_benchmark("Classification ROC-AUC (50k predictions)", "Scikit-Learn", rocAuc_fn, warmup=2, iterations=5))
    results.append(run_benchmark("OneHotEncoder fitTransform (50k rows)", "Scikit-Learn", ohe_fn, warmup=2, iterations=5))
    results.append(run_benchmark("NaiveBayesClassifier fit (1k×100, 3 classes)", "Scikit-Learn", nb_fn, warmup=2, iterations=10))
    results.append(run_benchmark("KernelSHAP Explain (5 feats, 100 coalitions)", "SHAP", shap_fn, warmup=2, iterations=5))
    print()
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SwiftSci Python Benchmark Suite v3.5.0")
    parser.add_argument("--json", metavar="PATH", help="Export results to JSON file")
    parser.add_argument("--suite", help="Filter by suite name (stats, dataframe, ml, forecast, extensions)")
    parser.add_argument("--filter", help="Filter by benchmark name substring")
    parser.add_argument("--rounds", type=int, default=3, help="Number of measurement rounds")
    parser.add_argument("--iterations", type=int, default=7, help="Iterations per round")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    args = parser.parse_args()

    BenchmarkConfig.default_rounds = args.rounds
    BenchmarkConfig.default_iterations = args.iterations
    BenchmarkConfig.default_warmup = args.warmup

    print("╔══════════════════════════════════════════════════════════╗")
    print("║        SwiftSci Python Benchmark Suite — v3.5.0          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Platform   : {platform.machine()} ({platform.system()})")
    print(f"Python     : {sys.version.split()[0]}")
    print(f"NumPy      : {np.__version__}")
    print(f"Pandas     : {pd.__version__}")
    print(f"Config     : {args.rounds} rounds × {args.iterations} iters (warmup={args.warmup})")
    if args.suite:
        print(f"Suite      : \"{args.suite}\"")
    if args.filter:
        print(f"Filter     : \"{args.filter}\"")
    print("")

    suites = [
        ("stats", bench_stats),
        ("dataframe", bench_dataframe),
        ("ml", bench_ml),
        ("forecast", bench_forecast),
        ("extensions", bench_extensions),
    ]

    all_results = []
    for suite_name, fn in suites:
        if args.suite and args.suite.lower() not in suite_name.lower():
            continue
        res = fn()
        if args.filter:
            res = [r for r in res if args.filter.lower() in r["name"].lower()]
        all_results.extend(res)

    # Formatted summary table
    print("─" * 135)
    print(f"{'Benchmark':<52} {'Module':<18} {'Mean ± 95% CI (ms)':<26} {'Trimmed(ms)':<12} {'Median(ms)':<12} {'Min..Max (ms)':<18} {'RAM(MB)':<8}")
    print("─" * 135)
    for r in all_results:
        ci_str = f"{r['meanMs']:8.3f} ± {r['marginOfError95Ms']:5.3f}"
        min_max_str = f"{r['minMs']:.3f}..{r['maxMs']:.3f}"
        print(f"{r['name']:<52} {r['module']:<18} {ci_str:<26} {r['trimmedMeanMs']:<12.3f} {r['medianMs']:<12.3f} {min_max_str:<18} {r['memoryMB']:<8.1f}")
    print("─" * 135)

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
