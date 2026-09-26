#!/usr/bin/env python3
"""
generate_fixtures.py — Deterministic fixture generator for SwiftSci Benchmarks.

Generates identical binary arrays and CSV files for both Swift and Python
benchmark runners, ensuring true byte-level parity (Apple-to-Apple).

Output Directory: Benchmarks/Data/
"""

import os
import sys
import hashlib
import json
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def write_f64_bin(filename: str, array: np.ndarray) -> str:
    path = os.path.join(DATA_DIR, filename)
    # Ensure little-endian IEEE-754 double precision
    arr_f64 = np.ascontiguousarray(array, dtype="<f8")
    with open(path, "wb") as f:
        f.write(arr_f64.tobytes())
    return path

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    ensure_dir(DATA_DIR)
    print(f"Generating benchmark fixtures in: {DATA_DIR}")
    checksums = {}

    # ──────────────────────────────────────────────────────────────────────────
    # 1. SwiftStats Fixtures
    # ──────────────────────────────────────────────────────────────────────────
    print("  • Generating stats_1m.bin (1,000,000 f64) ...")
    rng = np.random.default_rng(seed=42)
    stats_1m = rng.uniform(-50.0, 50.0, size=1_000_000)
    p = write_f64_bin("stats_1m.bin", stats_1m)
    checksums["stats_1m.bin"] = sha256_file(p)

    print("  • Generating correlation 500k pairs (stats_500k_a.bin, stats_500k_b.bin) ...")
    rng_a = np.random.default_rng(seed=42)
    stats_500k_a = rng_a.uniform(-50.0, 50.0, size=500_000)
    p = write_f64_bin("stats_500k_a.bin", stats_500k_a)
    checksums["stats_500k_a.bin"] = sha256_file(p)

    rng_b = np.random.default_rng(seed=99)
    stats_500k_b = rng_b.uniform(-50.0, 50.0, size=500_000)
    p = write_f64_bin("stats_500k_b.bin", stats_500k_b)
    checksums["stats_500k_b.bin"] = sha256_file(p)

    print("  • Generating hypothesis testing 100k pairs (stats_100k_a.bin, stats_100k_b.bin) ...")
    rng_101 = np.random.default_rng(seed=101)
    stats_100k_a = rng_101.uniform(-50.0, 50.0, size=100_000)
    p = write_f64_bin("stats_100k_a.bin", stats_100k_a)
    checksums["stats_100k_a.bin"] = sha256_file(p)

    rng_202 = np.random.default_rng(seed=202)
    stats_100k_b = rng_202.uniform(-50.0, 50.0, size=100_000)
    p = write_f64_bin("stats_100k_b.bin", stats_100k_b)
    checksums["stats_100k_b.bin"] = sha256_file(p)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. SwiftDataFrame Fixtures
    # ──────────────────────────────────────────────────────────────────────────
    print("  • Generating dataframe_100k.csv (100,000 rows × 5 cols) ...")
    n_df = 100_000
    df_rng = np.random.default_rng(seed=42)
    cats = df_rng.choice(["alpha", "beta", "gamma", "delta"], size=n_df)
    val_a = df_rng.uniform(0.0, 100.0, size=n_df)
    val_b = df_rng.uniform(0.0, 50.0, size=n_df)
    flags = [("true" if i % 2 == 0 else "false") for i in range(n_df)]

    df = pd.DataFrame({
        "id": np.arange(n_df),
        "category": cats,
        "value_a": np.round(val_a, 4),
        "value_b": np.round(val_b, 4),
        "flag": flags,
    })
    csv_path = os.path.join(DATA_DIR, "dataframe_100k.csv")
    df.to_csv(csv_path, index=False)
    checksums["dataframe_100k.csv"] = sha256_file(csv_path)

    print("  • Generating dataframe_join_100k.csv (100,000 rows for hash join) ...")
    df_join = pd.DataFrame({
        "id": np.arange(n_df),
        "weight": np.round(np.arange(n_df) * 0.05, 4)
    })
    csv_join_path = os.path.join(DATA_DIR, "dataframe_join_100k.csv")
    df_join.to_csv(csv_join_path, index=False)
    checksums["dataframe_join_100k.csv"] = sha256_file(csv_join_path)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. SwiftML & SwiftCluster Fixtures
    # ──────────────────────────────────────────────────────────────────────────
    print("  • Generating regression_10k_10 (X: 10,000×10, y: 10,000) ...")
    reg_rng = np.random.default_rng(seed=42)
    X_reg = reg_rng.uniform(-5.0, 5.0, size=(10_000, 10))
    weights = np.arange(1.0, 11.0)
    y_reg = X_reg @ weights + 1.0 + reg_rng.uniform(-0.1, 0.1, size=10_000)
    p_rx = write_f64_bin("regression_10k_10_X.bin", X_reg)
    p_ry = write_f64_bin("regression_10k_10_y.bin", y_reg)
    checksums["regression_10k_10_X.bin"] = sha256_file(p_rx)
    checksums["regression_10k_10_y.bin"] = sha256_file(p_ry)

    print("  • Generating classification_1k_4 (X: 1,000×4, y: 1,000) ...")
    cls_rng = np.random.default_rng(seed=42)
    X_cls = cls_rng.uniform(-5.0, 5.0, size=(1_000, 4))
    y_cls = np.where((X_cls[:, 0] > 0) & (X_cls[:, 1] > 0), 0.0, 1.0)
    p_cx = write_f64_bin("classification_1k_4_X.bin", X_cls)
    p_cy = write_f64_bin("classification_1k_4_y.bin", y_cls)
    checksums["classification_1k_4_X.bin"] = sha256_file(p_cx)
    checksums["classification_1k_4_y.bin"] = sha256_file(p_cy)

    print("  • Generating kmeans_10k_4.bin (10,000×4 f64) ...")
    km_rng = np.random.default_rng(seed=42)
    X_km = km_rng.uniform(-10.0, 10.0, size=(10_000, 4))
    p_km = write_f64_bin("kmeans_10k_4.bin", X_km)
    checksums["kmeans_10k_4.bin"] = sha256_file(p_km)

    print("  • Generating pca_1k_100.bin (1,000×100 f64) ...")
    pca_rng = np.random.default_rng(seed=77)
    X_pca = pca_rng.uniform(-10.0, 10.0, size=(1_000, 100))
    p_pca = write_f64_bin("pca_1k_100.bin", X_pca)
    checksums["pca_1k_100.bin"] = sha256_file(p_pca)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. SwiftForecast Fixtures
    # ──────────────────────────────────────────────────────────────────────────
    print("  • Generating forecast_50k.bin (50,000 f64 time series) ...")
    fc_rng = np.random.default_rng(seed=42)
    t = np.arange(50_000, dtype=np.float64)
    # Trend + seasonality (period=12) + noise
    series_fc = 20.0 + t * 0.005 + 8.0 * np.sin(t * 2.0 * np.pi / 12.0) + fc_rng.uniform(-1.0, 1.0, size=50_000)
    p_fc = write_f64_bin("forecast_50k.bin", series_fc)
    checksums["forecast_50k.bin"] = sha256_file(p_fc)

    # Save manifest
    manifest_path = os.path.join(DATA_DIR, "checksums.json")
    with open(manifest_path, "w") as f:
        json.dump(checksums, f, indent=2)
    print(f"\n✅ All fixtures generated successfully. Manifest saved to: {manifest_path}\n")

if __name__ == "__main__":
    main()
