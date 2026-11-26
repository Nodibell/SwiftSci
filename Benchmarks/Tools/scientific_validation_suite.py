"""Scientific validation and accuracy-gated benchmark suite for SwiftSci.

Executes isolated SwiftSci (`SwiftSciBenchmarkWorker --scientific-suite`) and
Python reference (`Benchmarks/Python/scientific_reference_worker.py`) subprocesses
in alternating engine order, separating fit vs inference/forecast stages, process
Peak RSS (`ru_maxrss`) from operation heap, flagging `< 1000 ns` resolution-limited
measurements, and enforcing accuracy/finiteness/error gates prior to speedups.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import execution_policy  # noqa: E402
import runner  # noqa: E402

RESOLUTION_THRESHOLD_NS = 1000.0

AIR_PASSENGERS_144 = [
    112.0, 118.0, 132.0, 129.0, 121.0, 135.0, 148.0, 148.0, 136.0, 119.0, 104.0, 118.0,
    115.0, 126.0, 141.0, 135.0, 125.0, 149.0, 170.0, 170.0, 158.0, 133.0, 114.0, 140.0,
    145.0, 150.0, 178.0, 163.0, 172.0, 178.0, 199.0, 199.0, 184.0, 162.0, 146.0, 166.0,
    171.0, 180.0, 193.0, 181.0, 183.0, 218.0, 230.0, 242.0, 209.0, 191.0, 172.0, 194.0,
    196.0, 196.0, 236.0, 235.0, 229.0, 243.0, 264.0, 272.0, 237.0, 211.0, 180.0, 201.0,
    204.0, 188.0, 235.0, 227.0, 234.0, 264.0, 302.0, 293.0, 259.0, 229.0, 203.0, 229.0,
    242.0, 233.0, 267.0, 269.0, 270.0, 315.0, 364.0, 347.0, 312.0, 274.0, 237.0, 278.0,
    284.0, 277.0, 317.0, 313.0, 318.0, 374.0, 413.0, 405.0, 355.0, 306.0, 271.0, 306.0,
    315.0, 301.0, 356.0, 348.0, 355.0, 422.0, 465.0, 467.0, 404.0, 347.0, 305.0, 336.0,
    340.0, 318.0, 362.0, 348.0, 363.0, 435.0, 491.0, 505.0, 404.0, 359.0, 310.0, 337.0,
    360.0, 342.0, 406.0, 396.0, 420.0, 472.0, 548.0, 559.0, 463.0, 407.0, 362.0, 405.0,
    417.0, 391.0, 419.0, 461.0, 472.0, 535.0, 622.0, 606.0, 508.0, 461.0, 390.0, 432.0,
]


class _LCG:
    """Deterministic cross-language linear congruential PRNG (64-bit Knuth multiplier)."""

    def __init__(self, seed=42):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_unit(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return ((self.state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def next_normal(self, std=1.0):
        u1 = max(1e-12, self.next_unit())
        u2 = self.next_unit()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2) * std


def all_finite(values):
    """Return True if all nested floats are finite and non-empty when sequence."""
    if isinstance(values, (int, float)):
        return math.isfinite(float(values))
    if isinstance(values, (list, tuple)):
        return all(all_finite(v) for v in values)
    return True


def median(values):
    """Compute exact standard median (no trimmed means)."""
    if not values:
        raise ValueError("Cannot compute median of empty sequence")
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def quantile(ordered, q):
    """Linearly interpolated empirical quantile on pre-sorted values."""
    if not ordered:
        raise ValueError("Cannot compute quantile of empty sequence")
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = min(len(ordered) - 1, lo + 1)
    frac = pos - lo
    return float(ordered[lo]) * (1.0 - frac) + float(ordered[hi]) * frac


def summarize_samples(samples_ns, resolution_threshold_ns=RESOLUTION_THRESHOLD_NS):
    """Summarize single-process timing samples with median, IQR, min, max, and clock check."""
    vals = [float(x) for x in samples_ns]
    if not vals:
        return None
    ordered = sorted(vals)
    med = median(ordered)
    q1 = quantile(ordered, 0.25)
    q3 = quantile(ordered, 0.75)
    return {
        "count": len(vals),
        "raw_samples_ns": [int(x) for x in samples_ns],
        "median_ns": med,
        "q1_ns": q1,
        "q3_ns": q3,
        "iqr_ns": q3 - q1,
        "min_ns": ordered[0],
        "max_ns": ordered[-1],
        "resolution_limited": med < resolution_threshold_ns,
    }


def aggregate_process_runs(process_sample_lists, resolution_threshold_ns=RESOLUTION_THRESHOLD_NS):
    """Aggregate independent process runs via median-of-medians + IQR/min/max."""
    per_proc_medians = []
    valid_raw = []
    for samples in process_sample_lists:
        if samples:
            per_proc_medians.append(median(samples))
            valid_raw.append([int(x) for x in samples])
    if not per_proc_medians:
        return None
    ordered = sorted(per_proc_medians)
    med_of_med = median(ordered)
    q1 = quantile(ordered, 0.25)
    q3 = quantile(ordered, 0.75)
    return {
        "process_count": len(per_proc_medians),
        "raw_process_medians_ns": per_proc_medians,
        "raw_process_samples_ns": valid_raw,
        "median_of_medians_ns": med_of_med,
        "median_of_medians_ms": med_of_med / 1_000_000.0,
        "q1_ns": q1,
        "q3_ns": q3,
        "iqr_ns": q3 - q1,
        "iqr_ms": (q3 - q1) / 1_000_000.0,
        "min_ns": ordered[0],
        "max_ns": ordered[-1],
        "resolution_limited": med_of_med < resolution_threshold_ns,
    }


def compute_rss_summary(process_rss_bytes):
    """Summarize whole-process Peak RSS (`ru_maxrss`); distinct from operation heap footprint."""
    valid = [float(b) for b in process_rss_bytes if b is not None and b > 0]
    if not valid:
        return None
    ordered = sorted(valid)
    med = median(ordered)
    q1 = quantile(ordered, 0.25)
    q3 = quantile(ordered, 0.75)
    mib = 1024.0 * 1024.0
    return {
        "metric_scope": "whole_process_peak_rss_ru_maxrss",
        "raw_bytes": [int(b) for b in valid],
        "median_bytes": med,
        "median_mib": med / mib,
        "iqr_bytes": q3 - q1,
        "iqr_mib": (q3 - q1) / mib,
        "min_bytes": ordered[0],
        "max_bytes": ordered[-1],
    }


def regression_metrics(y_true, y_pred):
    """Compute finite-checked MAE, RMSE, sMAPE, WAPE, R^2, and max absolute error."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {"valid": False, "all_finite": False}
    if not (all_finite(y_true) and all_finite(y_pred)):
        return {"valid": False, "all_finite": False}
    n = len(y_true)
    abs_errs = [abs(a - b) for a, b in zip(y_true, y_pred)]
    sq_errs = [(a - b) ** 2 for a, b in zip(y_true, y_pred)]
    mae = sum(abs_errs) / n
    rmse = math.sqrt(sum(sq_errs) / n)
    sum_abs_actual = sum(abs(a) for a in y_true)
    wape = (sum(abs_errs) / sum_abs_actual) if sum_abs_actual > 1e-12 else None

    smape_terms = []
    smape_zero_denoms = 0
    for a, b in zip(y_true, y_pred):
        denom = abs(a) + abs(b)
        if denom > 1e-12:
            smape_terms.append(2.0 * abs(a - b) / denom)
        else:
            smape_zero_denoms += 1
    smape = (sum(smape_terms) / len(smape_terms)) if smape_terms else None

    mean_y = sum(y_true) / n
    ss_tot = sum((a - mean_y) ** 2 for a in y_true)
    ss_res = sum(sq_errs)
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else None

    return {
        "valid": True,
        "all_finite": True,
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "smape": smape,
        "smape_zero_denom_count": smape_zero_denoms,
        "r2": r2,
        "max_abs_err": max(abs_errs),
    }


def forecast_metrics(train_series, y_true, y_pred, period=12):
    """Compute forecasting accuracy metrics including MASE and seasonal-naive comparison."""
    base = regression_metrics(y_true, y_pred)
    if not base.get("valid"):
        return base
    m = period if len(train_series) > period else 1
    naive_scale_terms = [
        abs(train_series[i] - train_series[i - m]) for i in range(m, len(train_series))
    ]
    naive_scale = (
        sum(naive_scale_terms) / len(naive_scale_terms) if naive_scale_terms else 0.0
    )
    mase = (base["mae"] / naive_scale) if naive_scale > 1e-12 else None

    # Matched seasonal-naive (or persistence) baseline over horizon
    h = len(y_true)
    snaive_preds = [
        float(train_series[len(train_series) - m + (i % m)]) for i in range(h)
    ]
    snaive_eval = regression_metrics(y_true, snaive_preds)
    base["mase"] = mase
    base["mase_period"] = m
    base["mase_zero_scale"] = naive_scale <= 1e-12
    base["seasonal_naive_rmse"] = snaive_eval.get("rmse")
    base["seasonal_naive_mae"] = snaive_eval.get("mae")
    return base


def classification_metrics(y_true, y_pred, probabilities=None):
    """Compute accuracy, macro-F1, and confusion counts."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {"valid": False, "all_finite": False}
    if not (all_finite(y_true) and all_finite(y_pred)):
        return {"valid": False, "all_finite": False}
    yt = [int(round(v)) for v in y_true]
    yp = [int(round(v)) for v in y_pred]
    labels = sorted(set(yt) | set(yp))
    correct = sum(1 for a, b in zip(yt, yp) if a == b)
    acc = correct / len(yt)
    f1s = []
    confusion = {str(lbl): {str(plbl): 0 for plbl in labels} for lbl in labels}
    for a, b in zip(yt, yp):
        confusion[str(a)][str(b)] += 1
    for lbl in labels:
        tp = confusion[str(lbl)][str(lbl)]
        fp = sum(confusion[str(other)][str(lbl)] for other in labels if other != lbl)
        fn = sum(confusion[str(lbl)][str(other)] for other in labels if other != lbl)
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    probs_finite = all_finite(probabilities) if probabilities is not None else True
    return {
        "valid": probs_finite,
        "all_finite": probs_finite,
        "accuracy": acc,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "confusion": confusion,
    }


def verify_g037_alignment(series, fitted_values, residuals, tol=1e-10):
    """Verify G-037 closed-loop fittedValues contract and Saura residual alignment identity."""
    if fitted_values is None or residuals is None:
        return {"applicable": True, "passed": False, "reason": "Missing fittedValues or residuals"}
    if not (all_finite(fitted_values) and all_finite(residuals)):
        return {"applicable": True, "passed": False, "reason": "Non-finite fittedValues or residuals"}
    if len(fitted_values) != len(series):
        return {
            "applicable": True,
            "passed": False,
            "reason": f"Length mismatch: fitted={len(fitted_values)} vs series={len(series)}",
        }
    warm_up = len(series) - len(residuals)
    if warm_up < 0:
        return {"applicable": True, "passed": False, "reason": f"Negative warmUp={warm_up}"}
    max_warmup_diff = max(
        (abs(fitted_values[t] - series[t]) for t in range(warm_up)), default=0.0
    )
    max_identity_diff = max(
        (
            abs((series[t] - residuals[t - warm_up]) - fitted_values[t])
            for t in range(warm_up, len(series))
        ),
        default=0.0,
    )
    passed = max_warmup_diff <= tol and max_identity_diff <= tol
    return {
        "applicable": True,
        "passed": passed,
        "warm_up": warm_up,
        "fitted_count": len(fitted_values),
        "residuals_count": len(residuals),
        "max_warmup_diff": max_warmup_diff,
        "max_identity_diff": max_identity_diff,
        "saura_formula_exact_match": max_identity_diff <= tol,
    }


def verify_treeshap_exact_coalitions(
    flat_trees, instances, shap_matrix, base_values, predictions, num_features, tol=1e-9
):
    """Independent 2^K coalition Shapley verifier matching uniform leaf-count TreeSHAP weights."""
    if not flat_trees or not instances or not shap_matrix:
        return {"applicable": False, "passed": False, "reason": "Missing TreeSHAP artifacts"}

    def subtree_weights(tree):
        memo = {}

        def w(idx):
            if idx in memo:
                return memo[idx]
            if idx < 0 or idx >= len(tree):
                return 0.0
            node = tree[idx]
            is_leaf = node[5] >= 0.5
            if is_leaf:
                memo[idx] = 1.0
                return 1.0
            total = w(int(round(node[2]))) + w(int(round(node[3])))
            memo[idx] = total
            return total

        return [w(i) for i in range(len(tree))]

    def expected_tree_value(tree, weights):
        def exp_node(idx):
            node = tree[idx]
            if node[5] >= 0.5:
                return float(node[4])
            l_idx, r_idx = int(round(node[2])), int(round(node[3]))
            w_l = weights[l_idx] if 0 <= l_idx < len(tree) else 0.0
            w_r = weights[r_idx] if 0 <= r_idx < len(tree) else 0.0
            tot = w_l + w_r
            if tot <= 0.0:
                return float(node[4])
            return (w_l / tot) * exp_node(l_idx) + (w_r / tot) * exp_node(r_idx)

        return exp_node(0)

    def predict_tree(tree, row):
        idx = 0
        while 0 <= idx < len(tree):
            node = tree[idx]
            if node[5] >= 0.5:
                return float(node[4])
            f_idx = int(round(node[0]))
            idx = int(round(node[2])) if row[f_idx] <= node[1] else int(round(node[3]))
        return 0.0

    def exact_tree_shap(tree, row):
        weights = subtree_weights(tree)
        active = sorted(
            {
                int(round(n[0]))
                for n in tree
                if n[5] < 0.5 and 0 <= int(round(n[0])) < num_features
            }
        )
        k = len(active)
        if k > 12:
            raise ValueError(f"Tree active feature count K={k} exceeds exact coalition cap 12")
        phi = [0.0] * num_features
        if k == 0:
            return phi, expected_tree_value(tree, weights), predict_tree(tree, row)
        feat_bit = {f: b for b, f in enumerate(active)}

        def eval_subset(mask):
            def trav(idx):
                node = tree[idx]
                if node[5] >= 0.5:
                    return float(node[4])
                f_idx = int(round(node[0]))
                bit = feat_bit.get(f_idx)
                l_idx, r_idx = int(round(node[2])), int(round(node[3]))
                if bit is not None and (mask & (1 << bit)) != 0:
                    return trav(l_idx if row[f_idx] <= node[1] else r_idx)
                w_l, w_r = weights[l_idx], weights[r_idx]
                tot = w_l + w_r
                return (w_l / tot) * trav(l_idx) + (w_r / tot) * trav(r_idx)

            return trav(0)

        v = [eval_subset(mask) for mask in range(1 << k)]
        fact = [math.factorial(i) for i in range(k + 1)]
        for i, feat in enumerate(active):
            bit = 1 << i
            acc = 0.0
            for mask in range(1 << k):
                if (mask & bit) != 0:
                    continue
                s = mask.bit_count()
                weight = (fact[s] * fact[k - s - 1]) / fact[k]
                acc += weight * (v[mask | bit] - v[mask])
            phi[feat] = acc
        return phi, expected_tree_value(tree, weights), predict_tree(tree, row)

    max_shap_err = 0.0
    max_additivity_err = 0.0
    n_trees = float(len(flat_trees))
    for inst_idx, row in enumerate(instances):
        oracle_phi = [0.0] * num_features
        oracle_base = 0.0
        oracle_pred = 0.0
        for tree in flat_trees:
            t_phi, t_base, t_pred = exact_tree_shap(tree, row)
            for j in range(num_features):
                oracle_phi[j] += t_phi[j] / n_trees
            oracle_base += t_base / n_trees
            oracle_pred += t_pred / n_trees

        got_phi = shap_matrix[inst_idx]
        got_base = base_values[inst_idx]
        got_pred = predictions[inst_idx]
        max_additivity_err = max(
            max_additivity_err, abs((sum(got_phi) + got_base) - got_pred)
        )
        max_shap_err = max(
            max_shap_err,
            abs(got_base - oracle_base),
            abs(got_pred - oracle_pred),
            *(abs(a - b) for a, b in zip(got_phi, oracle_phi)),
        )

    passed = max_shap_err <= tol and max_additivity_err <= tol
    return {
        "applicable": True,
        "passed": passed,
        "instances_verified": len(instances),
        "trees_verified": len(flat_trees),
        "max_coalition_phi_diff": max_shap_err,
        "max_additivity_diff": max_additivity_err,
    }


def build_datasets():
    """Generate all deterministic datasets shared identically across Swift and Python."""
    rng = _LCG(seed=42)
    syn500 = [
        20.0 + 0.25 * t + 8.0 * math.sin(2.0 * math.pi * t / 12.0) + rng.next_normal(1.2)
        for t in range(500)
    ]
    rng_50k = _LCG(seed=1042)
    syn50k = [
        20.0 + 0.005 * t + 8.0 * math.sin(2.0 * math.pi * t / 12.0) + rng_50k.next_normal(1.2)
        for t in range(50000)
    ]

    def _generate_arima_dgp(seed, n_pts):
        """Generate (1 - 0.6B)(1 - B) y_t = (1 - 0.35B) e_t stochastic ARIMA(1,1,1) series."""
        rng_dgp = _LCG(seed=seed)
        eps = [rng_dgp.next_normal(1.0) for _ in range(n_pts)]
        w = [0.0] * n_pts
        for t in range(1, n_pts):
            w[t] = 0.6 * w[t - 1] + eps[t] - 0.35 * eps[t - 1]
        y = [100.0] * n_pts
        for t in range(1, n_pts):
            y[t] = y[t - 1] + w[t]
        return y

    arima_dgp_syn500 = _generate_arima_dgp(seed=2026, n_pts=500)
    arima_dgp_syn50k = _generate_arima_dgp(seed=2026, n_pts=50000)

    zero_var_1000 = [42.0] * 1000

    # Anomaly detection datasets (with injected ground-truth outlier indices)
    rng_a1 = _LCG(seed=42)
    anom144 = [
        100.0 + 10.0 * math.sin(2.0 * math.pi * t / 12.0) + rng_a1.next_normal(0.5)
        for t in range(144)
    ]
    anom144_truth = {30, 75, 120}
    anom144[30] += 18.0
    anom144[75] -= 17.0
    anom144[120] += 20.0

    rng_a2 = _LCG(seed=777)
    anom500_stationary = [
        100.0 + 12.0 * math.sin(2.0 * math.pi * t / 12.0) + rng_a2.next_normal(0.5)
        for t in range(500)
    ]
    anom500_truth = {45, 130, 250, 360, 440}
    for idx, delta in ((45, 22.0), (130, -21.0), (250, 25.0), (360, -23.0), (440, 24.0)):
        anom500_stationary[idx] += delta

    rng_a3 = _LCG(seed=778)
    anom500_trended = [
        50.0 + 0.05 * t + 12.0 * math.sin(2.0 * math.pi * t / 12.0) + rng_a3.next_normal(0.5)
        for t in range(500)
    ]
    for idx, delta in ((45, 22.0), (130, -21.0), (250, 25.0), (360, -23.0), (440, 24.0)):
        anom500_trended[idx] += delta

    # Regression split (N=2000, p=10)
    rng_reg = _LCG(seed=101)
    weights_reg = [2.5, -1.8, 0.9, 3.2, -0.7, 1.4, -2.1, 0.5, 1.1, -1.3]
    x_reg = [[rng_reg.next_normal(1.0) for _ in range(10)] for _ in range(2000)]
    y_reg = [
        3.0 + sum(w_j * x_i[j] for j, w_j in enumerate(weights_reg)) + rng_reg.next_normal(0.25)
        for x_i in x_reg
    ]

    # Nonlinear tree regression split (N=2000, p=10)
    y_tree_reg = [
        2.0 * x_i[0]
        + 1.5 * math.sin(x_i[1])
        + 1.2 * (x_i[2] * x_i[3])
        - 0.8 * (x_i[4] ** 2)
        + rng_reg.next_normal(0.2)
        for x_i in x_reg
    ]

    # Binary classification split (N=2000, p=10)
    rng_cls = _LCG(seed=303)
    w_cls = [1.4, -1.2, 0.9, -0.8, 1.1, -0.6, 0.4, -0.3, 0.7, -0.5]
    x_cls = [[rng_cls.next_normal(1.0) for _ in range(10)] for _ in range(2000)]
    y_cls = [
        1.0 if (sum(w_j * x_i[j] for j, w_j in enumerate(w_cls)) + rng_cls.next_normal(0.35)) > 0 else 0.0
        for x_i in x_cls
    ]

    # PCA & KMeans dataset (N=1500, p=12, 4 well-separated clusters)
    rng_km = _LCG(seed=505)
    centers = [
        [4.0, 0.0] * 6,
        [-4.0, 0.0] * 6,
        [0.0, 4.0] * 6,
        [0.0, -4.0] * 6,
    ]
    x_cluster = []
    for idx in range(1500):
        c = centers[idx % 4]
        x_cluster.append([c[j] + rng_km.next_normal(0.6) for j in range(12)])

    # TreeSHAP exact K=4, K=8, and MC K=16 datasets
    def make_tree_shap_data(seed, n_rows, n_feat):
        r = _LCG(seed=seed)
        x = [[r.next_normal(1.0) for _ in range(n_feat)] for _ in range(n_rows)]
        if n_feat <= 8:
            y = [
                3.0 * row[0] - 2.0 * row[1] + 1.5 * row[2] + (row[3] if n_feat > 3 else 0.0)
                + r.next_normal(0.15)
                for row in x
            ]
        else:
            y = [
                sum((-1.0 if j % 2 else 1.0) * (1.0 + 0.35 * (j % 4)) * row[j] for j in range(n_feat))
                + r.next_normal(0.1)
                for row in x
            ]
        return x[:400], y[:400], x[400:425]

    x_shap_k4_tr, y_shap_k4_tr, x_shap_k4_inst = make_tree_shap_data(601, 450, 4)
    x_shap_k8_tr, y_shap_k8_tr, x_shap_k8_inst = make_tree_shap_data(602, 450, 8)
    x_shap_k16_tr, y_shap_k16_tr, x_shap_k16_inst = make_tree_shap_data(603, 500, 16)

    rng_stats = _LCG(seed=808)
    stats_100k = [rng_stats.next_normal(1.0) + 0.1 * (i % 5) for i in range(100000)]

    return {
        "syn500": syn500,
        "syn50k": syn50k,
        "arima_dgp_syn500": arima_dgp_syn500,
        "arima_dgp_syn50k": arima_dgp_syn50k,
        "airpassengers_144": AIR_PASSENGERS_144,
        "zero_var_1000": zero_var_1000,
        "anom144": anom144,
        "anom144_truth": sorted(anom144_truth),
        "anom500": anom500_stationary,
        "anom500_trended": anom500_trended,
        "anom500_truth": sorted(anom500_truth),
        "reg": (x_reg[:1600], y_reg[:1600], x_reg[1600:], y_reg[1600:]),
        "tree_reg": (x_reg[:1600], y_tree_reg[:1600], x_reg[1600:], y_tree_reg[1600:]),
        "cls": (x_cls[:1600], y_cls[:1600], x_cls[1600:], y_cls[1600:]),
        "cluster": x_cluster,
        "shap_k4": (x_shap_k4_tr, y_shap_k4_tr, x_shap_k4_inst),
        "shap_k8": (x_shap_k8_tr, y_shap_k8_tr, x_shap_k8_inst),
        "shap_k16": (x_shap_k16_tr, y_shap_k16_tr, x_shap_k16_inst),
        "stats_5": [1.0, 2.0, 3.0, 4.0, 5.0],
        "stats_100k": stats_100k,
    }


def evaluate_case_gate(spec, swift_resp, py_resp=None, fit_swift=None, infer_swift=None, fit_py=None, infer_py=None):
    """Enforce finiteness, error contract, accuracy, and estimator comparability before speedups."""
    expect_error = bool(spec.get("expect_error", False))
    comparable_estimator = bool(spec.get("comparable_estimator", True))
    has_python = py_resp is not None and not spec.get("swift_only", False)

    reasons = []
    passed = True
    metrics_out = {}

    # 1. Fail-closed expected-error gate
    if expect_error:
        err_occ = bool(swift_resp.get("error_occurred", False))
        is_fitted = swift_resp.get("is_fitted", True)
        if not (err_occ and is_fitted is False):
            passed = False
            reasons.append(f"Expected fail-closed error with is_fitted=False (got err={err_occ}, is_fitted={is_fitted})")
        return {
            "passed": passed,
            "expect_error_verified": passed,
            "comparable_estimator": False,
            "reasons": reasons,
            "metrics": {
                "swift_error_occurred": err_occ,
                "swift_error_type": swift_resp.get("error_type"),
                "swift_is_fitted": is_fitted,
                "python_converged": (not py_resp.get("error_occurred", False)) if py_resp else None,
            },
            "speedup_fit": {"eligible": False, "reason": "EXCLUDED_EXPECTED_ERROR"},
            "speedup_infer": {"eligible": False, "reason": "EXCLUDED_EXPECTED_ERROR"},
        }

    # Unexpected worker error
    if swift_resp.get("error_occurred"):
        passed = False
        reasons.append(f"Unexpected Swift error: {swift_resp.get('error_message')}")
    if has_python and py_resp.get("error_occurred"):
        passed = False
        reasons.append(f"Unexpected Python error: {py_resp.get('error_message')}")

    # 2. Kind-specific accuracy & scientific verification gates
    kind = spec["gate_kind"]

    if kind == "forecast":
        train = spec["train_series"]
        actual = spec["actual"]
        period = spec.get("period", 12)
        sw_fc = swift_resp.get("predictions") or []
        sw_m = forecast_metrics(train, actual, sw_fc, period=period)
        metrics_out["swift"] = sw_m
        if not sw_m.get("valid"):
            passed = False
            reasons.append("Swift forecast invalid or non-finite")
        elif spec.get("max_rmse") is not None and sw_m["rmse"] > spec["max_rmse"]:
            passed = False
            reasons.append(f"Swift RMSE {sw_m['rmse']:.4f} > max {spec['max_rmse']}")
        if spec.get("max_mase") is not None and sw_m.get("mase") is not None and sw_m["mase"] > spec["max_mase"]:
            passed = False
            reasons.append(f"Swift MASE {sw_m['mase']:.4f} > max {spec['max_mase']}")
        if spec.get("beat_seasonal_naive_rmse") and sw_m.get("valid") and sw_m.get("seasonal_naive_rmse") is not None and sw_m["rmse"] >= sw_m["seasonal_naive_rmse"]:
            passed = False
            reasons.append(f"Swift RMSE {sw_m['rmse']:.4f} >= seasonal naive {sw_m['seasonal_naive_rmse']:.4f}")

        if spec.get("verify_g037"):
            g037 = verify_g037_alignment(
                train,
                swift_resp.get("fitted_values"),
                swift_resp.get("residuals"),
            )
            metrics_out["g037"] = g037
            if not g037["passed"]:
                passed = False
                reasons.append(f"G-037 alignment failed: {g037.get('reason') or g037}")
            if spec.get("check_late_1960_scale"):
                fv = swift_resp.get("fitted_values") or []
                late_err = max((abs(fv[t] - train[t]) for t in range(139, 144)), default=999.0)
                metrics_out["g037"]["late_1960_max_err"] = late_err
                if late_err >= 50.0:
                    passed = False
                    reasons.append(f"Late-1960 G-037 fitted scale drifted: max_err={late_err:.2f}")

        if has_python:
            py_fc = py_resp.get("predictions") or []
            py_m = forecast_metrics(train, actual, py_fc, period=period)
            metrics_out["python"] = py_m
            if not py_m.get("valid"):
                passed = False
                reasons.append("Python forecast invalid or non-finite")
            else:
                if spec.get("max_rmse_gap_vs_python") is not None and sw_m.get("valid"):
                    rmse_gap = sw_m["rmse"] - py_m["rmse"]
                    metrics_out["rmse_gap_vs_python"] = rmse_gap
                    if rmse_gap > spec["max_rmse_gap_vs_python"]:
                        passed = False
                        reasons.append(f"Swift RMSE exceeds Python by {rmse_gap:.4f} > {spec['max_rmse_gap_vs_python']}")
                if comparable_estimator and spec.get("max_fc_parity_diff") is not None:
                    diff = max((abs(a - b) for a, b in zip(sw_fc, py_fc)), default=0.0)
                    metrics_out["forecast_max_abs_diff_vs_python"] = diff
                    if diff > spec["max_fc_parity_diff"]:
                        passed = False
                        reasons.append(f"Forecast parity diff {diff:.4f} > {spec['max_fc_parity_diff']}")

    elif kind == "zero_variance":
        sw_fc = swift_resp.get("predictions") or []
        best_order = swift_resp.get("best_order")
        metrics_out["best_order"] = best_order
        metrics_out["predictions_finite"] = all_finite(sw_fc)
        metrics_out["max_abs_err_from_42"] = max((abs(v - 42.0) for v in sw_fc), default=999.0)
        if not sw_fc or not metrics_out["predictions_finite"] or metrics_out["max_abs_err_from_42"] > 1e-6:
            passed = False
            reasons.append("Zero-variance AutoARIMA output invalid or drifted from 42.0")
        if best_order is not None and best_order[:3] != [0, 0, 0]:
            passed = False
            reasons.append(f"Expected AutoARIMA order [0,0,0,...], got {best_order}")

    elif kind == "poly_check":
        flags = swift_resp.get("bool_flags")
        expected = spec["expected_flags"]
        metrics_out["bool_flags"] = flags
        metrics_out["expected_flags"] = expected
        if flags != expected:
            passed = False
            reasons.append(f"Polynomial stability flags {flags} != expected {expected}")

    elif kind == "anomaly":
        pred_set = set(swift_resp.get("anomaly_indices") or [])
        truth_set = set(spec["truth_indices"])
        tp = len(pred_set & truth_set)
        fp = len(pred_set - truth_set)
        fn = len(truth_set - pred_set)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        metrics_out["anomaly"] = {
            "predicted_indices": sorted(pred_set),
            "truth_indices": sorted(truth_set),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "threshold": swift_resp.get("anomaly_threshold"),
        }
        if f1 < spec.get("min_f1", 0.75):
            passed = False
            reasons.append(f"Anomaly F1 {f1:.3f} < {spec.get('min_f1', 0.75)}")
        if spec.get("min_recall") is not None and rec < spec["min_recall"]:
            passed = False
            reasons.append(f"Anomaly recall {rec:.3f} < {spec['min_recall']}")

    elif kind == "supervised_reg":
        y_true = spec["actual"]
        sw_p = swift_resp.get("predictions") or []
        sw_m = regression_metrics(y_true, sw_p)
        metrics_out["swift"] = sw_m
        if not sw_m.get("valid") or (sw_m.get("r2") is not None and sw_m["r2"] < spec.get("min_r2", 0.5)):
            passed = False
            reasons.append(f"Swift regression quality below floor: {sw_m}")
        if has_python:
            py_p = py_resp.get("predictions") or []
            py_m = regression_metrics(y_true, py_p)
            metrics_out["python"] = py_m
            if not py_m.get("valid"):
                passed = False
                reasons.append("Python regression invalid")
            if (
                spec.get("max_r2_gap_vs_python") is not None
                and sw_m.get("r2") is not None
                and py_m.get("r2") is not None
            ):
                r2_gap = py_m["r2"] - sw_m["r2"]
                metrics_out["r2_gap_vs_python"] = r2_gap
                if r2_gap > spec["max_r2_gap_vs_python"]:
                    passed = False
                    reasons.append(f"Regression R^2 trails Python by {r2_gap:.4f} > {spec['max_r2_gap_vs_python']}")
            if spec.get("max_pred_diff") is not None and len(sw_p) == len(py_p):
                diff = max((abs(a - b) for a, b in zip(sw_p, py_p)), default=0.0)
                metrics_out["max_abs_pred_diff"] = diff
                if diff > spec["max_pred_diff"]:
                    passed = False
                    reasons.append(f"Prediction parity diff {diff:.6g} > {spec['max_pred_diff']}")

    elif kind == "supervised_cls":
        y_true = spec["actual"]
        sw_p = swift_resp.get("predictions") or []
        sw_m = classification_metrics(y_true, sw_p, swift_resp.get("probabilities"))
        metrics_out["swift"] = sw_m
        if not sw_m.get("valid") or sw_m["accuracy"] < spec.get("min_accuracy", 0.80):
            passed = False
            reasons.append(f"Swift classifier accuracy {sw_m.get('accuracy')} < {spec.get('min_accuracy', 0.80)}")
        if has_python:
            py_p = py_resp.get("predictions") or []
            py_m = classification_metrics(y_true, py_p, py_resp.get("probabilities"))
            metrics_out["python"] = py_m
            acc_diff = abs(sw_m["accuracy"] - py_m["accuracy"]) if sw_m.get("valid") and py_m.get("valid") else 1.0
            metrics_out["accuracy_gap_vs_python"] = acc_diff
            if acc_diff > spec.get("max_acc_gap", 0.05):
                passed = False
                reasons.append(f"Classifier accuracy gap {acc_diff:.4f} > {spec.get('max_acc_gap', 0.05)}")

    elif kind == "pca":
        sw_evr = (swift_resp.get("metrics") or {}).get("explained_variance_ratio_sum")
        py_evr = (py_resp.get("metrics") or {}).get("explained_variance_ratio_sum") if py_resp else None
        sw_mat = swift_resp.get("matrix_output") or []
        metrics_out["swift"] = {
            "total_evr": sw_evr,
            "all_finite": all_finite(sw_mat),
            "shape": [len(sw_mat), len(sw_mat[0]) if sw_mat else 0],
        }
        if not metrics_out["swift"]["all_finite"] or sw_evr is None:
            passed = False
            reasons.append("PCA output invalid")
        if has_python and py_evr is not None:
            metrics_out["python"] = {"total_evr": py_evr}
            evr_diff = abs(sw_evr - py_evr)
            metrics_out["total_evr_abs_diff"] = evr_diff
            if evr_diff > spec.get("max_evr_diff", 1e-4):
                passed = False
                reasons.append(f"PCA EVR diff {evr_diff:.6g} > {spec.get('max_evr_diff', 1e-4)}")

    elif kind == "kmeans":
        sw_wcss = (swift_resp.get("metrics") or {}).get("wcss")
        py_wcss = (py_resp.get("metrics") or {}).get("wcss") if py_resp else None
        sw_assign = swift_resp.get("predictions") or []
        metrics_out["swift"] = {"wcss": sw_wcss, "all_finite": all_finite(sw_assign), "n": len(sw_assign)}
        if not metrics_out["swift"]["all_finite"] or sw_wcss is None or sw_wcss <= 0:
            passed = False
            reasons.append("KMeans invalid output")
        if has_python and py_wcss is not None:
            metrics_out["python"] = {"wcss": py_wcss}
            rel_gap = abs(sw_wcss - py_wcss) / max(1e-12, py_wcss)
            metrics_out["wcss_rel_gap"] = rel_gap
            if rel_gap > spec.get("max_wcss_rel_gap", 0.05):
                passed = False
                reasons.append(f"KMeans WCSS rel gap {rel_gap:.4f} > {spec.get('max_wcss_rel_gap', 0.05)}")

    elif kind == "treeshap":
        mat = swift_resp.get("matrix_output") or []
        bases = swift_resp.get("shap_base_values") or []
        preds = swift_resp.get("shap_predictions") or []
        exact_flag = (swift_resp.get("bool_flags") or [False])[0]
        max_add = max(
            (abs((sum(row) + b) - p) for row, b, p in zip(mat, bases, preds)),
            default=999.0,
        )
        metrics_out["swift"] = {
            "exact_flag": exact_flag,
            "all_finite": all_finite(mat) and all_finite(bases) and all_finite(preds),
            "max_additivity_diff": max_add,
        }
        if not metrics_out["swift"]["all_finite"] or max_add > 1e-8:
            passed = False
            reasons.append(f"TreeSHAP additivity failed: max_diff={max_add}")
        if spec.get("expect_exact") is not None and exact_flag != spec["expect_exact"]:
            passed = False
            reasons.append(f"TreeSHAP exact_flag={exact_flag} != expected {spec['expect_exact']}")
        if spec.get("verify_coalitions"):
            coal = verify_treeshap_exact_coalitions(
                swift_resp.get("flat_trees"),
                spec["instances"],
                mat,
                bases,
                preds,
                spec["num_features"],
            )
            metrics_out["coalition_oracle"] = coal
            if not coal.get("passed"):
                passed = False
                reasons.append(f"Independent 2^K coalition verification failed: {coal}")
        if has_python and py_resp is not None:
            py_mat = py_resp.get("matrix_output") or []
            metrics_out["python"] = {"all_finite": all_finite(py_mat)}

    elif kind == "stats_ddof1":
        m = swift_resp.get("metrics") or {}
        skew = m.get("skewness")
        kurt = m.get("kurtosis")
        py_m = (py_resp.get("metrics") or {}) if py_resp else {}
        py_skew = py_m.get("skew_bias_true")
        py_kurt = py_m.get("kurt_bias_true")
        n = spec["n"]
        scale_skew = ((n - 1.0) / n) ** 1.5
        scale_kurt = ((n - 1.0) / n) ** 2
        expected_skew = py_skew * scale_skew if py_skew is not None else None
        expected_kurt = (py_kurt + 3.0) * scale_kurt - 3.0 if py_kurt is not None else None
        skew_diff = abs(skew - expected_skew) if (skew is not None and expected_skew is not None) else 999.0
        kurt_diff = abs(kurt - expected_kurt) if (kurt is not None and expected_kurt is not None) else 999.0
        metrics_out["swift"] = {"skewness_ddof1": skew, "kurtosis_ddof1": kurt}
        metrics_out["python"] = {"scipy_skew_bias_true": py_skew, "scipy_kurtosis_bias_true": py_kurt}
        metrics_out["ddof1_relation"] = {
            "n": n,
            "expected_skew_from_scipy": expected_skew,
            "expected_kurt_from_scipy": expected_kurt,
            "skew_diff": skew_diff,
            "kurt_diff": kurt_diff,
        }
        if skew is None or kurt is None or skew_diff > 1e-9 or kurt_diff > 1e-9:
            passed = False
            reasons.append(f"Stats ddof=1 verification failed: skew_diff={skew_diff}, kurt_diff={kurt_diff}")

    elif kind == "swiftllm_ablation":
        m = swift_resp.get("metrics") or {}
        max_diff = m.get("max_abs_logit_diff", 999.0)
        metrics_out["swiftllm"] = m
        if not math.isfinite(max_diff) or max_diff >= 1e-4:
            passed = False
            reasons.append(f"SwiftLLM KV-cache vs full-forward logit parity failed: diff={max_diff}")

    # 3. Compute gated speedups ONLY if all correctness & comparability gates pass
    def stage_speedup(sw_summary, py_summary):
        if spec.get("internal_ablation"):
            # Internal SwiftSci comparison: full_forward (fit_summary) / kvcache (infer_summary)
            return None
        if not has_python or py_summary is None or sw_summary is None:
            return {"eligible": False, "reason": "NO_PYTHON_BASELINE"}
        if not passed:
            return {"eligible": False, "reason": "ACCURACY_OR_FINITENESS_GATE_FAILED"}
        if not comparable_estimator:
            return {"eligible": False, "reason": "NON_COMPARABLE_ESTIMATOR_OR_WEIGHTS"}
        if sw_summary.get("resolution_limited") or py_summary.get("resolution_limited"):
            return {"eligible": False, "reason": "EXCLUDED_RESOLUTION_LIMITED_LT_1000NS"}
        sw_ns = sw_summary["median_of_medians_ns"]
        py_ns = py_summary["median_of_medians_ns"]
        ratio = py_ns / sw_ns if sw_ns > 0 else None
        return {
            "eligible": True,
            "swift_vs_python_ratio": ratio,
            "faster_engine": "SwiftSci" if (ratio and ratio > 1.0) else "Python",
            "factor": ratio if (ratio and ratio >= 1.0) else (1.0 / ratio if ratio else None),
        }

    sp_fit = stage_speedup(fit_swift, fit_py)
    sp_infer = stage_speedup(infer_swift, infer_py)
    if spec.get("internal_ablation") and fit_swift and infer_swift:
        if not passed:
            sp_infer = {"eligible": False, "reason": "LOGIT_PARITY_GATE_FAILED"}
        elif fit_swift.get("resolution_limited") or infer_swift.get("resolution_limited"):
            sp_infer = {"eligible": False, "reason": "EXCLUDED_RESOLUTION_LIMITED_LT_1000NS"}
        else:
            ratio = fit_swift["median_of_medians_ns"] / infer_swift["median_of_medians_ns"]
            sp_infer = {
                "eligible": True,
                "kvcache_vs_full_forward_speedup": ratio,
                "full_forward_ms": fit_swift["median_of_medians_ms"],
                "kvcache_ms": infer_swift["median_of_medians_ms"],
            }
        sp_fit = {"eligible": False, "reason": "INTERNAL_ABLATION_BASELINE_STAGE"}

    return {
        "passed": passed,
        "expect_error_verified": False,
        "comparable_estimator": comparable_estimator,
        "reasons": reasons,
        "metrics": metrics_out,
        "speedup_fit": sp_fit,
        "speedup_infer": sp_infer,
    }


def _invoke_worker(cmd, request_payload, env, timeout_s=180):
    with tempfile.TemporaryDirectory(prefix="swiftsci_val_") as tmp:
        req_path = Path(tmp) / "req.json"
        res_path = Path(tmp) / "res.json"
        req_path.write_text(json.dumps(request_payload), encoding="utf-8")
        proc = subprocess.run(
            [*cmd, str(req_path), str(res_path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Worker failed ({proc.returncode}): {' '.join(cmd)}\nSTDERR: {proc.stderr}\nSTDOUT: {proc.stdout}"
            )
        return json.loads(res_path.read_text(encoding="utf-8"))


def build_suite_cases(datasets, processes=5, warmups=2, samples=5):
    """Build ordered list of benchmark and scientific validation workload definitions."""
    syn500 = datasets["syn500"]
    syn50k = datasets["syn50k"]
    dgp500 = datasets["arima_dgp_syn500"]
    dgp50k = datasets["arima_dgp_syn50k"]
    ap144 = datasets["airpassengers_144"]
    x_reg_tr, y_reg_tr, x_reg_te, y_reg_te = datasets["reg"]
    _, y_tree_tr, _, y_tree_te = datasets["tree_reg"]
    x_cls_tr, y_cls_tr, x_cls_te, y_cls_te = datasets["cls"]

    base_req = {"schema_version": 1, "warmups": warmups, "samples": samples}
    cases = []

    def add(case_id, **kwargs):
        kwargs["workload_id"] = case_id
        kwargs.setdefault("processes", processes)
        cases.append(kwargs)

    # --- Group A: Forecasting & G-033 / G-037 ---
    add(
        "ts_hw_syn500_opt",
        group="SwiftForecast",
        title="Holt-Winters Additive (N=500, h=24, SwiftSci Nelder-Mead alpha vs Statsmodels optimized=True)",
        request={**base_req, "workload_id": "ts_hw_syn500_opt", "operation": "holt_winters", "series": syn500[:476], "horizon": 24, "period": 12, "beta": 0.1, "gamma": 0.1},
        gate_kind="forecast",
        train_series=syn500[:476],
        actual=syn500[476:],
        period=12,
        comparable_estimator=False,
        disclosure="SwiftSci optimizes alpha with fixed beta=0.1, gamma=0.1; Statsmodels jointly optimizes (alpha, beta, gamma, initial states). Compare forecast accuracy directly; exclude raw optimization speedup.",
    )
    add(
        "ts_hw_syn500_fixed",
        group="SwiftForecast",
        title="Holt-Winters Additive Fixed Parameters (alpha=0.3, beta=0.1, gamma=0.1, N=500, h=24)",
        request={**base_req, "workload_id": "ts_hw_syn500_fixed", "operation": "holt_winters", "series": syn500[:476], "horizon": 24, "period": 12, "alpha": 0.3, "beta": 0.1, "gamma": 0.1},
        gate_kind="forecast",
        train_series=syn500[:476],
        actual=syn500[476:],
        period=12,
        comparable_estimator=True,
        max_rmse=4.0,
        max_mase=1.5,
    )
    add(
        "ts_hw_syn50k_fixed",
        group="SwiftForecast",
        title="Holt-Winters Additive Fixed Parameters (alpha=0.3, beta=0.1, gamma=0.1, N=50k, h=24)",
        request={**base_req, "workload_id": "ts_hw_syn50k_fixed", "operation": "holt_winters", "series": syn50k[:49976], "horizon": 24, "period": 12, "alpha": 0.3, "beta": 0.1, "gamma": 0.1},
        gate_kind="forecast",
        train_series=syn50k[:49976],
        actual=syn50k[49976:],
        period=12,
        comparable_estimator=True,
        max_rmse=4.0,
    )
    add(
        "ts_arima_syn500_matched_hr",
        group="SwiftForecast",
        title="ARIMA(1,1,1) Matched Hannan-Rissanen (N=500, h=24)",
        request={**base_req, "workload_id": "ts_arima_syn500_matched_hr", "operation": "arima_fit_forecast", "series": syn500[:476], "horizon": 24, "p": 1, "d": 1, "q": 1, "arima_method": "hannan_rissanen"},
        gate_kind="forecast",
        train_series=syn500[:476],
        actual=syn500[476:],
        period=12,
        verify_g037=True,
        comparable_estimator=True,
        disclosure="Matched estimator family: SwiftSci ARIMA .hannanRissanen vs Statsmodels method='hannan_rissanen'.",
    )
    add(
        "ts_arima_syn500_vs_statespace",
        group="SwiftForecast",
        title="ARIMA(1,1,1) Hannan-Rissanen vs Statsmodels Default Statespace MLE (N=500, h=24)",
        request={**base_req, "workload_id": "ts_arima_syn500_vs_statespace", "operation": "arima_fit_forecast", "series": syn500[:476], "horizon": 24, "p": 1, "d": 1, "q": 1, "arima_method": "statespace"},
        gate_kind="forecast",
        train_series=syn500[:476],
        actual=syn500[476:],
        period=12,
        verify_g037=True,
        comparable_estimator=False,
        disclosure="Non-comparable estimators (Hannan-Rissanen two-stage OLS vs Kalman statespace exact MLE): raw speedup excluded by protocol.",
    )
    add(
        "ts_arima_syn50k_vs_statespace",
        group="SwiftForecast",
        title="ARIMA(1,1,1) on Sinusoidal syn50k (Hannan-Rissanen vs Statsmodels Statespace MLE; Py HR rejected non-invertible MA)",
        request={**base_req, "workload_id": "ts_arima_syn50k_vs_statespace", "operation": "arima_fit_forecast", "series": syn50k[:49976], "horizon": 24, "p": 1, "d": 1, "q": 1, "arima_method": "statespace"},
        gate_kind="forecast",
        train_series=syn50k[:49976],
        actual=syn50k[49976:],
        period=12,
        verify_g037=True,
        comparable_estimator=False,
        disclosure="Statsmodels method='hannan_rissanen' rejects non-invertible MA on deterministic 50k sinusoid; compared against Statespace MLE with speedup excluded.",
    )
    add(
        "ts_arima_dgp_syn50k_matched_hr",
        group="SwiftForecast",
        title="ARIMA(1,1,1) on Pure ARIMA(1,1,1) Stochastic DGP (Matched Hannan-Rissanen, N=50k, h=24)",
        request={**base_req, "workload_id": "ts_arima_dgp_syn50k_matched_hr", "operation": "arima_fit_forecast", "series": dgp50k[:49976], "horizon": 24, "p": 1, "d": 1, "q": 1, "arima_method": "hannan_rissanen"},
        gate_kind="forecast",
        train_series=dgp50k[:49976],
        actual=dgp50k[49976:],
        period=1,
        verify_g037=True,
        comparable_estimator=True,
        disclosure="Matched Hannan-Rissanen estimator family on well-posed N=50k stochastic ARIMA(1,1,1) DGP.",
    )
    add(
        "ts_arima_dgp_syn500_matched_hr",
        group="SwiftForecast",
        title="ARIMA(1,1,1) on Pure ARIMA(1,1,1) Stochastic DGP (Matched Hannan-Rissanen, N=500, h=24)",
        request={**base_req, "workload_id": "ts_arima_dgp_syn500_matched_hr", "operation": "arima_fit_forecast", "series": dgp500[:476], "horizon": 24, "p": 1, "d": 1, "q": 1, "arima_method": "hannan_rissanen"},
        gate_kind="forecast",
        train_series=dgp500[:476],
        actual=dgp500[476:],
        period=1,
        verify_g037=True,
        comparable_estimator=True,
    )
    add(
        "ts_sarima_expect_fail_syn500",
        group="SwiftForecast",
        title="SARIMA(1,1,1)(1,0,0)[12] Sinusoidal Edge Fixture (Expected Fail-Closed in v3.13.2, N=500)",
        request={**base_req, "workload_id": "ts_sarima_expect_fail_syn500", "operation": "sarima_fit_forecast", "series": syn500[:476], "horizon": 24, "p": 1, "d": 1, "q": 1, "P": 1, "D": 0, "Q": 0, "s": 12, "expect_error": True},
        gate_kind="expect_error",
        expect_error=True,
        comparable_estimator=False,
        disclosure="Deterministically sinusoidal fixture triggers near-unit-root AR/MA instability in linear LS SARIMA; v3.13.2 fails closed (isFitted=false) rather than emitting NaN. Excluded from speedups.",
    )
    add(
        "ts_sarima_expect_fail_syn50k",
        group="SwiftForecast",
        title="SARIMA(1,1,1)(1,0,0)[12] Sinusoidal Edge Fixture (Expected Fail-Closed in v3.13.2, N=50k)",
        request={**base_req, "workload_id": "ts_sarima_expect_fail_syn50k", "operation": "sarima_fit_forecast", "series": syn50k[:49976], "horizon": 24, "p": 1, "d": 1, "q": 1, "P": 1, "D": 0, "Q": 0, "s": 12, "expect_error": True},
        gate_kind="expect_error",
        expect_error=True,
        comparable_estimator=False,
        disclosure="N=50k sinusoidal fixture fails closed in v3.13.2 without NaN emission. Excluded from speedups.",
    )
    add(
        "ts_sarima_airpassengers_110_010",
        group="SwiftForecast",
        title="SARIMA(1,1,0)(0,1,0)[12] AirPassengers (train=120, h=24) + G-037 Verification",
        request={**base_req, "workload_id": "ts_sarima_airpassengers_110_010", "operation": "sarima_fit_forecast", "series": ap144[:120], "horizon": 24, "p": 1, "d": 1, "q": 0, "P": 0, "D": 1, "Q": 0, "s": 12},
        gate_kind="forecast",
        train_series=ap144[:120],
        actual=ap144[120:],
        period=12,
        verify_g037=True,
        beat_seasonal_naive_rmse=True,
        max_rmse_gap_vs_python=5.0,
        comparable_estimator=False,
        disclosure="SwiftSci linear/LS SARIMA vs Statsmodels SARIMAX Kalman MLE (different solvers; out-of-sample RMSE beats seasonal naive & matches Statsmodels within 5.0 + G-037 alignment validated).",
    )
    add(
        "ts_sarima_airpassengers_011_011",
        group="SwiftForecast",
        title="SARIMA(0,1,1)(0,1,1)[12] Canonical Airline Model AirPassengers (train=120, h=24) + G-037",
        request={**base_req, "workload_id": "ts_sarima_airpassengers_011_011", "operation": "sarima_fit_forecast", "series": ap144[:120], "horizon": 24, "p": 0, "d": 1, "q": 1, "P": 0, "D": 1, "Q": 1, "s": 12},
        gate_kind="forecast",
        train_series=ap144[:120],
        actual=ap144[120:],
        period=12,
        verify_g037=True,
        beat_seasonal_naive_rmse=True,
        max_rmse_gap_vs_python=5.0,
        comparable_estimator=False,
        disclosure="SwiftSci linear/LS SARIMA vs Statsmodels SARIMAX Kalman MLE (different solvers; out-of-sample RMSE beats seasonal naive & matches Statsmodels within 5.0 + G-037 alignment validated).",
    )
    add(
        "ts_sarima_airpassengers_144_g037",
        group="SwiftForecast",
        title="SARIMA(1,1,0)(0,1,0)[12] Full AirPassengers 144-pt G-037 In-Sample Closed-Loop Audit",
        request={**base_req, "workload_id": "ts_sarima_airpassengers_144_g037", "operation": "sarima_fit_forecast", "series": ap144, "horizon": 12, "p": 1, "d": 1, "q": 0, "P": 0, "D": 1, "Q": 0, "s": 12},
        gate_kind="forecast",
        train_series=ap144,
        actual=ap144[-12:],
        period=12,
        verify_g037=True,
        check_late_1960_scale=True,
        swift_only=True,
        comparable_estimator=False,
        processes=3,
    )
    add(
        "ts_auto_arima_valid",
        group="SwiftForecast",
        title="AutoARIMA Non-Seasonal Grid Selection on ARIMA(1,1,1) DGP (N=500, h=24)",
        request={**base_req, "workload_id": "ts_auto_arima_valid", "operation": "auto_arima_fit_forecast", "series": dgp500[:476], "horizon": 24, "max_p": 2, "max_d": 1, "max_q": 2, "seasonal": False, "max_candidates": 18},
        gate_kind="forecast",
        train_series=dgp500[:476],
        actual=dgp500[476:],
        period=1,
        verify_g037=True,
        swift_only=True,
        comparable_estimator=False,
    )
    add(
        "ts_auto_arima_zero_variance",
        group="SwiftForecast",
        title="AutoARIMA Constant Zero-Variance Series (N=1000 -> (0,0,0), G-033)",
        request={**base_req, "workload_id": "ts_auto_arima_zero_variance", "operation": "auto_arima_fit_forecast", "series": datasets["zero_var_1000"], "horizon": 12, "max_p": 2, "max_d": 1, "max_q": 2, "seasonal": False, "max_candidates": 12},
        gate_kind="zero_variance",
        swift_only=True,
        comparable_estimator=False,
        processes=3,
    )
    add(
        "ts_auto_arima_poly_sign_check",
        group="SwiftForecast",
        title="AutoARIMA Polynomial Stability & MA Sign Convention Check (G-033b)",
        request={**base_req, "workload_id": "ts_auto_arima_poly_sign_check", "operation": "auto_arima_polynomial_check", "coefficients": [0.5, -0.3]},
        gate_kind="poly_check",
        expected_flags=[True, True],
        swift_only=True,
        comparable_estimator=False,
        processes=1,
    )
    add(
        "ts_auto_arima_poly_unstable_check",
        group="SwiftForecast",
        title="AutoARIMA Polynomial Unstable Rejection Check ([1.2, 0.0] -> false, G-033b)",
        request={**base_req, "workload_id": "ts_auto_arima_poly_unstable_check", "operation": "auto_arima_polynomial_check", "coefficients": [1.2, 0.0]},
        gate_kind="poly_check",
        expected_flags=[False, False],
        swift_only=True,
        comparable_estimator=False,
        processes=1,
    )

    # --- Group B: Anomaly Detection (SwiftSci-only truth-labeled) ---
    add(
        "anomaly_syn144_s04",
        group="SwiftForecast",
        title="TimeSeriesAnomalyDetector Truth-Labeled S-04 Series (N=144, 3 planted outliers)",
        request={**base_req, "workload_id": "anomaly_syn144_s04", "operation": "anomaly_detect", "series": datasets["anom144"], "period": 12, "max_anomalies_ratio": 0.10, "threshold_z": 3.0},
        gate_kind="anomaly",
        truth_indices=datasets["anom144_truth"],
        min_f1=0.80,
        swift_only=True,
        comparable_estimator=False,
        disclosure="Internal truth-labeled validation; no cross-library Python speedup baseline claimed.",
    )
    add(
        "anomaly_syn500_holdout",
        group="SwiftForecast",
        title="TimeSeriesAnomalyDetector Stationary Seasonal Holdout Series (N=500, 5 planted outliers)",
        request={**base_req, "workload_id": "anomaly_syn500_holdout", "operation": "anomaly_detect", "series": datasets["anom500"], "period": 12, "max_anomalies_ratio": 0.05, "threshold_z": 5.0},
        gate_kind="anomaly",
        truth_indices=datasets["anom500_truth"],
        min_f1=0.80,
        swift_only=True,
        comparable_estimator=False,
        disclosure="Stationary seasonal holdout with 5 planted outliers (+21..25 on sigma=0.5); threshold_z=5.0 prevents non-robust centered-MA trend bleed (~delta/12 ≈ 4 sigma) from flagging adjacent neighbors, alongside the trended diagnostic below.",
    )
    add(
        "anomaly_syn500_trended_diagnostic",
        group="SwiftForecast",
        title="TimeSeriesAnomalyDetector Trended Diagnostic (N=500, 5 planted outliers; edge medTrend limitation)",
        request={**base_req, "workload_id": "anomaly_syn500_trended_diagnostic", "operation": "anomaly_detect", "series": datasets["anom500_trended"], "period": 12, "max_anomalies_ratio": 0.05, "threshold_z": 3.0},
        gate_kind="anomaly",
        truth_indices=datasets["anom500_truth"],
        min_f1=0.0,
        min_recall=1.0,
        swift_only=True,
        comparable_estimator=False,
        processes=3,
        disclosure="Known v3.13.2 limitation documented: AnomalyDetection.swift replaces NaN edge trend in 0..<p/2 and n-p/2..<n with global medTrend, producing boundary false positives under linear trends while retaining recall=1.0.",
    )

    # --- Group C: Supervised ML & Unsupervised Clustering ---
    add(
        "ml_ols_regression",
        group="SwiftML",
        title="OLS LinearRegression (1600x10 train / 400x10 test)",
        request={**base_req, "workload_id": "ml_ols_regression", "operation": "ols_regression", "x_train": x_reg_tr, "y_train": y_reg_tr, "x_test": x_reg_te},
        gate_kind="supervised_reg",
        actual=y_reg_te,
        min_r2=0.99,
        max_pred_diff=1e-5,
        comparable_estimator=True,
    )
    add(
        "ml_logistic_regression",
        group="SwiftML",
        title="LogisticRegression (1600x10 train / 400x10 test)",
        request={**base_req, "workload_id": "ml_logistic_regression", "operation": "logistic_regression", "x_train": x_cls_tr, "y_train": y_cls_tr, "x_test": x_cls_te, "learning_rate": 0.1, "epochs": 300, "seed": 42},
        gate_kind="supervised_cls",
        actual=y_cls_te,
        min_accuracy=0.88,
        max_acc_gap=0.05,
        comparable_estimator=False,
        disclosure="Matched dataset/binary task; optimizer differs (SwiftSci GD vs Scikit-Learn L-BFGS + L2 C=1.0).",
    )
    add(
        "ml_linear_svc",
        group="SwiftML",
        title="LinearSVC (C=1.0, 1600x10 train / 400x10 test)",
        request={**base_req, "workload_id": "ml_linear_svc", "operation": "linear_svc", "x_train": x_cls_tr, "y_train": y_cls_tr, "x_test": x_cls_te, "c_param": 1.0, "seed": 42},
        gate_kind="supervised_cls",
        actual=y_cls_te,
        min_accuracy=0.88,
        max_acc_gap=0.05,
        comparable_estimator=True,
    )
    add(
        "ml_decision_tree_regressor",
        group="SwiftML",
        title="DecisionTreeRegressor (max_depth=5, 1600x10 train / 400x10 test)",
        request={**base_req, "workload_id": "ml_decision_tree_regressor", "operation": "decision_tree_regressor", "x_train": x_reg_tr, "y_train": y_tree_tr, "x_test": x_reg_te, "max_depth": 5, "seed": 42},
        gate_kind="supervised_reg",
        actual=y_tree_te,
        min_r2=0.55,
        comparable_estimator=True,
    )
    add(
        "ml_decision_tree_classifier",
        group="SwiftML",
        title="DecisionTreeClassifier (Gini, max_depth=5, 1600x10 train / 400x10 test)",
        request={**base_req, "workload_id": "ml_decision_tree_classifier", "operation": "decision_tree_classifier", "x_train": x_cls_tr, "y_train": y_cls_tr, "x_test": x_cls_te, "max_depth": 5, "seed": 42},
        gate_kind="supervised_cls",
        actual=y_cls_te,
        min_accuracy=0.72,
        max_acc_gap=0.03,
        comparable_estimator=True,
    )
    add(
        "ml_random_forest_regressor",
        group="SwiftML",
        title="RandomForestRegressor (n_estimators=50, max_depth=6, 1600x10 train / 400x10 test)",
        request={**base_req, "workload_id": "ml_random_forest_regressor", "operation": "random_forest_regressor", "x_train": x_reg_tr, "y_train": y_tree_tr, "x_test": x_reg_te, "n_estimators": 50, "max_depth": 6, "seed": 42},
        gate_kind="supervised_reg",
        actual=y_tree_te,
        min_r2=0.55,
        max_r2_gap_vs_python=0.08,
        comparable_estimator=True,
    )
    add(
        "ml_random_forest_classifier",
        group="SwiftML",
        title="RandomForestClassifier (n_estimators=50, max_depth=6, 1600x10 train / 400x10 test)",
        request={**base_req, "workload_id": "ml_random_forest_classifier", "operation": "random_forest_classifier", "x_train": x_cls_tr, "y_train": y_cls_tr, "x_test": x_cls_te, "n_estimators": 50, "max_depth": 6, "seed": 42},
        gate_kind="supervised_cls",
        actual=y_cls_te,
        min_accuracy=0.83,
        max_acc_gap=0.06,
        comparable_estimator=True,
    )
    add(
        "ml_gbdt_regressor",
        group="SwiftML",
        title="GradientBoostedTreesRegressor (n_estimators=50, max_depth=3, lr=0.1, 1600x10)",
        request={**base_req, "workload_id": "ml_gbdt_regressor", "operation": "gbdt_regressor", "x_train": x_reg_tr, "y_train": y_tree_tr, "x_test": x_reg_te, "n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "seed": 42},
        gate_kind="supervised_reg",
        actual=y_tree_te,
        min_r2=0.70,
        max_r2_gap_vs_python=0.05,
        comparable_estimator=True,
    )
    add(
        "ml_hist_gbdt_regressor",
        group="SwiftML",
        title="HistGradientBoostingRegressor (n_estimators=50, max_depth=5, bins=255, 1600x10)",
        request={**base_req, "workload_id": "ml_hist_gbdt_regressor", "operation": "hist_gbdt_regressor", "x_train": x_reg_tr, "y_train": y_tree_tr, "x_test": x_reg_te, "n_estimators": 50, "max_depth": 5, "max_bins": 255, "learning_rate": 0.1, "seed": 42},
        gate_kind="supervised_reg",
        actual=y_tree_te,
        min_r2=0.75,
        max_r2_gap_vs_python=0.05,
        comparable_estimator=True,
    )
    add(
        "ml_hist_gbdt_classifier",
        group="SwiftML",
        title="HistGradientBoostingClassifier (n_estimators=50, max_depth=5, bins=255, 1600x10)",
        request={**base_req, "workload_id": "ml_hist_gbdt_classifier", "operation": "hist_gbdt_classifier", "x_train": x_cls_tr, "y_train": y_cls_tr, "x_test": x_cls_te, "n_estimators": 50, "max_depth": 5, "max_bins": 255, "learning_rate": 0.1, "seed": 42},
        gate_kind="supervised_cls",
        actual=y_cls_te,
        min_accuracy=0.84,
        max_acc_gap=0.06,
        comparable_estimator=True,
    )
    add(
        "cluster_pca",
        group="SwiftCluster",
        title="PCA SVD fit + transform (1500x12, n_components=4)",
        request={**base_req, "workload_id": "cluster_pca", "operation": "pca_fit_transform", "x_train": datasets["cluster"], "n_components": 4},
        gate_kind="pca",
        max_evr_diff=1e-4,
        comparable_estimator=True,
    )
    add(
        "cluster_kmeans",
        group="SwiftCluster",
        title="KMeans (n_clusters=4, 1500x12)",
        request={**base_req, "workload_id": "cluster_kmeans", "operation": "kmeans_fit_predict", "x_train": datasets["cluster"], "n_clusters": 4, "seed": 42},
        gate_kind="kmeans",
        max_wcss_rel_gap=0.05,
        comparable_estimator=True,
    )

    # --- Group D: TreeSHAP (`SwiftExplain`) ---
    x_k4, y_k4, inst_k4 = datasets["shap_k4"]
    add(
        "explain_treeshap_dt_k4",
        group="SwiftExplain",
        title="TreeSHAP Exact DecisionTreeRegressor (K<=4, 25 instances) + 2^K Coalition Oracle",
        request={**base_req, "workload_id": "explain_treeshap_dt_k4", "operation": "treeshap_decision_tree_regressor", "x_train": x_k4, "y_train": y_k4, "x_test": inst_k4, "instances": inst_k4, "max_depth": 3, "seed": 42},
        gate_kind="treeshap",
        expect_exact=True,
        verify_coalitions=True,
        instances=inst_k4,
        num_features=4,
        comparable_estimator=False,
        disclosure="SwiftSci uses exact 2^K enumeration with uniform leaf-count weights (1.0/leaf); SHAP C++ uses training-sample weights on its own sklearn tree. Independent 2^K coalition oracle verifies exact SwiftSci Shapley math.",
    )
    x_k8, y_k8, inst_k8 = datasets["shap_k8"]
    add(
        "explain_treeshap_rf_k8",
        group="SwiftExplain",
        title="TreeSHAP Exact RandomForestRegressor (10 trees, K<=8, 25 instances) + 2^K Coalition Oracle",
        request={**base_req, "workload_id": "explain_treeshap_rf_k8", "operation": "treeshap_random_forest_regressor", "x_train": x_k8, "y_train": y_k8, "x_test": inst_k8, "instances": inst_k8, "n_estimators": 10, "max_depth": 3, "seed": 42},
        gate_kind="treeshap",
        expect_exact=True,
        verify_coalitions=True,
        instances=inst_k8,
        num_features=8,
        comparable_estimator=False,
        disclosure="Uniform leaf-count coalition exact verification over serialized forest; raw timing vs SHAP C++ disclosed with differing tree weighting semantics.",
    )
    x_k16, y_k16, inst_k16 = datasets["shap_k16"]
    add(
        "explain_treeshap_dt_k16_mc",
        group="SwiftExplain",
        title="TreeSHAP Monte Carlo Regime DecisionTreeRegressor (p=16, max_depth=6, 25 instances)",
        request={**base_req, "workload_id": "explain_treeshap_dt_k16_mc", "operation": "treeshap_decision_tree_regressor", "x_train": x_k16, "y_train": y_k16, "x_test": inst_k16, "instances": inst_k16, "max_depth": 6, "seed": 42},
        gate_kind="treeshap",
        expect_exact=False,
        verify_coalitions=False,
        instances=inst_k16,
        num_features=16,
        comparable_estimator=False,
        disclosure="Exercises Monte Carlo 2048-permutation fallback when active tree features K > 12 (maxExactFeatures).",
    )

    # --- Group E: SwiftStats (`ddof=1` convention) ---
    add(
        "stats_ddof1_small5",
        group="SwiftStats",
        title="Stats.skewness & Stats.kurtosis ([1,2,3,4,5], ddof=1 exact kurtosis = -1.912)",
        request={**base_req, "workload_id": "stats_ddof1_small5", "operation": "stats_skewness_kurtosis", "series": datasets["stats_5"]},
        gate_kind="stats_ddof1",
        n=5,
        comparable_estimator=False,
        processes=3,
        disclosure="5-element micro-op expected < 1000 ns (resolution-limited -> excluded from speedups); verifies exact ddof=1 formula vs scipy.stats bias=True * ((n-1)/n)^k.",
    )
    add(
        "stats_ddof1_100k",
        group="SwiftStats",
        title="Stats.skewness & Stats.kurtosis (N=100,000 continuous vector, ddof=1 relation)",
        request={**base_req, "workload_id": "stats_ddof1_100k", "operation": "stats_skewness_kurtosis", "series": datasets["stats_100k"]},
        gate_kind="stats_ddof1",
        n=100000,
        comparable_estimator=True,
    )

    # --- Group F: SwiftLLM KV-Cache Internal Ablation ---
    add(
        "llm_kvcache_ablation_p24_d16",
        group="SwiftLLM",
        title="SwiftLLM KV-Cache vs Full-Forward Decode Ablation (prompt=24, steps=16)",
        request={**base_req, "workload_id": "llm_kvcache_ablation_p24_d16", "operation": "swiftllm_kvcache_ablation", "prompt_len": 24, "decode_steps": 16},
        gate_kind="swiftllm_ablation",
        internal_ablation=True,
        swift_only=True,
        comparable_estimator=False,
        disclosure="Internal SwiftSci architectural ablation (no fake Python baseline). Requires max_abs_logit_diff < 1e-4.",
    )
    add(
        "llm_kvcache_ablation_p64_d24",
        group="SwiftLLM",
        title="SwiftLLM KV-Cache vs Full-Forward Decode Ablation (prompt=64, steps=24)",
        request={**base_req, "workload_id": "llm_kvcache_ablation_p64_d24", "operation": "swiftllm_kvcache_ablation", "prompt_len": 64, "decode_steps": 24},
        gate_kind="swiftllm_ablation",
        internal_ablation=True,
        swift_only=True,
        comparable_estimator=False,
        disclosure="Larger context internal KV-cache ablation with greedy logit parity check.",
    )

    return cases


def run_rolling_origin_checks(datasets, swift_worker, py_exe, py_worker, env):
    """Run deterministic 3-origin rolling-origin forecast comparisons (HW syn500, ARIMA DGP, AirPassengers)."""
    syn500 = datasets["syn500"]
    dgp500 = datasets["arima_dgp_syn500"]
    dgp50k = datasets["arima_dgp_syn50k"]
    ap144 = datasets["airpassengers_144"]
    origins = [
        ("hw_syn500_ro", "holt_winters", syn500, [428, 452, 476], 24, 12, {"alpha": 0.3, "beta": 0.1, "gamma": 0.1}),
        ("arima_dgp_ro", "arima_fit_forecast", dgp500, [428, 452, 476], 24, 1, {"p": 1, "d": 1, "q": 1, "arima_method": "hannan_rissanen"}),
        ("sarima_ap_ro", "sarima_fit_forecast", ap144, [108, 120, 132], 12, 12, {"p": 1, "d": 1, "q": 0, "P": 0, "D": 1, "Q": 0, "s": 12}),
    ]
    results = {}
    for key, op, series, splits, horizon, period, extra in origins:
        per_origin = []
        for idx, n_train in enumerate(splits):
            train = series[:n_train]
            actual = series[n_train : n_train + horizon]
            req = {
                "schema_version": 1,
                "workload_id": f"{key}_origin_{idx+1}",
                "operation": op,
                "warmups": 0,
                "samples": 1,
                "series": train,
                "horizon": horizon,
                "period": period,
                **extra,
            }
            sw_res = _invoke_worker([str(swift_worker), "--scientific-suite"], req, env)
            py_res = _invoke_worker([str(py_exe), str(py_worker)], req, env)
            sw_m = forecast_metrics(train, actual, sw_res.get("predictions") or [], period=period)
            py_m = forecast_metrics(train, actual, py_res.get("predictions") or [], period=period)
            g037 = (
                verify_g037_alignment(train, sw_res.get("fitted_values"), sw_res.get("residuals"))
                if op in ("arima_fit_forecast", "sarima_fit_forecast")
                else {"applicable": False, "passed": True}
            )
            per_origin.append({
                "origin_train_len": n_train,
                "horizon": horizon,
                "swift": sw_m,
                "python": py_m,
                "g037": g037,
            })
        passed = all(o["swift"].get("valid") and o["g037"]["passed"] for o in per_origin)
        results[key] = {
            "passed": passed,
            "origins": per_origin,
            "swift_mean_rmse": sum(o["swift"]["rmse"] for o in per_origin) / len(per_origin),
            "python_mean_rmse": sum(o["python"]["rmse"] for o in per_origin) / len(per_origin),
            "seasonal_naive_mean_rmse": sum(o["swift"]["seasonal_naive_rmse"] for o in per_origin) / len(per_origin),
        }
    return results


def execute_suite(
    root,
    swift_worker,
    python_engine,
    output_path,
    processes=5,
    warmups=2,
    samples=5,
    status="LOCAL",
    label="scientific-validation",
):
    """Execute the scientific validation suite and write an audited JSON report."""
    py_worker = root / "Benchmarks/Python/scientific_reference_worker.py"
    runner.verify_uninstrumented(swift_worker)
    metal_rec = runner.metal_build_record(swift_worker)
    src_id = runner.source_identity(root)
    host_env = runner.environment()
    power = runner.power_state()

    worker_env = execution_policy.worker_environment(execution_policy.LEGACY_MODE)
    # Inspect Python engine runtime once cleanly
    py_probe = subprocess.run(
        [
            str(python_engine),
            "-c",
            (
                "import json, sys, numpy, pandas, scipy, sklearn, statsmodels, shap; "
                f"sys.path.insert(0, {str(TOOLS_DIR)!r}); "
                "import execution_policy; "
                "print(json.dumps({"
                "'versions': {'python': sys.version.split()[0], 'numpy': numpy.__version__, "
                "'pandas': pandas.__version__, 'scipy': scipy.__version__, "
                "'scikit-learn': sklearn.__version__, 'statsmodels': statsmodels.__version__, "
                "'shap': shap.__version__}, "
                "'numerical_backend': execution_policy.numerical_backend()"
                "}))"
            ),
        ],
        env=worker_env,
        capture_output=True,
        text=True,
        check=True,
    )
    py_meta = json.loads(py_probe.stdout)

    datasets = build_datasets()
    cases = build_suite_cases(datasets, processes=processes, warmups=warmups, samples=samples)
    workload_reports = []

    for spec in cases:
        req = spec["request"]
        proc_count = spec["processes"]
        has_python = not spec.get("swift_only", False)
        engines = ["swiftsci", "python"] if has_python else ["swiftsci"]
        swift_runs = []
        py_runs = []

        for batch in range(proc_count):
            for eng in execution_policy.engine_order(engines, batch):
                if eng == "swiftsci":
                    swift_runs.append(_invoke_worker([str(swift_worker), "--scientific-suite"], req, worker_env))
                else:
                    py_runs.append(_invoke_worker([str(python_engine), str(py_worker)], req, worker_env))

        fit_swift = aggregate_process_runs([r.get("fit_samples_ns", []) for r in swift_runs])
        infer_swift = aggregate_process_runs([r.get("infer_samples_ns", []) for r in swift_runs])
        rss_swift = compute_rss_summary([r.get("peak_rss_bytes") for r in swift_runs])

        fit_py = aggregate_process_runs([r.get("fit_samples_ns", []) for r in py_runs]) if py_runs else None
        infer_py = aggregate_process_runs([r.get("infer_samples_ns", []) for r in py_runs]) if py_runs else None
        rss_py = compute_rss_summary([r.get("peak_rss_bytes") for r in py_runs]) if py_runs else None

        gate = evaluate_case_gate(
            spec,
            swift_runs[-1],
            py_runs[-1] if py_runs else None,
            fit_swift=fit_swift,
            infer_swift=infer_swift,
            fit_py=fit_py,
            infer_py=infer_py,
        )

        # Strip massive flat_trees array before archiving full case report
        report_entry = {
            "workload_id": spec["workload_id"],
            "group": spec["group"],
            "title": spec["title"],
            "expect_error": bool(spec.get("expect_error", False)),
            "comparable_estimator": bool(spec.get("comparable_estimator", True)),
            "swift_only": not has_python,
            "disclosure": spec.get("disclosure"),
            "swift_timing": {"fit": fit_swift, "infer": infer_swift},
            "python_timing": {"fit": fit_py, "infer": infer_py} if has_python else None,
            "swift_process_peak_rss": rss_swift,
            "python_process_peak_rss": rss_py,
            "gate": gate,
        }
        workload_reports.append(report_entry)

    rolling_origin = run_rolling_origin_checks(datasets, swift_worker, python_engine, py_worker, worker_env)

    all_passed = all(w["gate"]["passed"] for w in workload_reports) and all(
        r["passed"] for r in rolling_origin.values()
    )
    resolution_limited_count = sum(
        1
        for w in workload_reports
        for stage in ("fit", "infer")
        if (w["swift_timing"][stage] or {}).get("resolution_limited")
        or ((w["python_timing"] or {}).get(stage) or {}).get("resolution_limited")
    )

    payload = {
        "schema_version": 1,
        "status": status,
        "label": label,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_identity": {
            "commit": src_id["commit"],
            "dirty": src_id["dirty"],
            "dirty_entries": src_id["dirty_entries"],
            "tree_sha256": src_id["tree_sha256"],
        },
        "environment": host_env,
        "power_state": power,
        "swift_worker": {
            "path": str(swift_worker),
            "sha256": hashlib.sha256(swift_worker.read_bytes()).hexdigest(),
            "uninstrumented_verified": True,
            "metal_build_record": metal_rec,
        },
        "python_reference": {
            "executable": str(python_engine),
            **py_meta,
        },
        "execution_policy": {
            **execution_policy.policy(execution_policy.LEGACY_MODE),
            "processes": processes,
            "warmups": warmups,
            "samples": samples,
            "resolution_threshold_ns": RESOLUTION_THRESHOLD_NS,
        },
        "deferred_limitations": [
            {
                "id": "G-035",
                "status": "DEFERRED_LIMITATION",
                "summary": "TreeSHAP for GradientBoostedTreesRegressor is unsupported and deferred past v3.13.2 (documented in DocC).",
            },
            {
                "id": "G-036",
                "status": "DEFERRED_LIMITATION",
                "summary": "Classifier TreeSHAP explains FlatTreeNode.value integer class labels with uniform leaf-count weights rather than class probabilities; deferred past v3.13.2 (documented in DocC).",
            },
        ],
        "rolling_origin_backtests": rolling_origin,
        "workloads": workload_reports,
        "summary": {
            "total_workloads": len(workload_reports),
            "passed_workloads": sum(1 for w in workload_reports if w["gate"]["passed"]),
            "failed_workloads": [w["workload_id"] for w in workload_reports if not w["gate"]["passed"]],
            "expect_error_verified": sum(1 for w in workload_reports if w["gate"]["expect_error_verified"]),
            "rolling_origin_passed": all(r["passed"] for r in rolling_origin.values()),
            "resolution_limited_stages": resolution_limited_count,
            "all_gates_passed": all_passed,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run SwiftSci scientific validation suite")
    parser.add_argument(
        "--swift-worker",
        type=Path,
        required=True,
        help="Path to Release SwiftSciBenchmarkWorker binary",
    )
    parser.add_argument(
        "--python-engine",
        type=Path,
        required=True,
        help="Path to Python interpreter with statsmodels/sklearn/shap",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON path",
    )
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--status",
        default="LOCAL",
        help="Run status label recorded in output metadata",
    )
    parser.add_argument(
        "--label",
        default="scientific-validation",
        help="Run identifier label recorded in output metadata",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    payload = execute_suite(
        root=root,
        swift_worker=args.swift_worker.resolve(),
        # Preserve the venv executable path; resolving its symlink bypasses pyvenv.cfg.
        python_engine=Path(os.path.abspath(str(args.python_engine))),
        output_path=args.output.resolve(),
        processes=args.processes,
        warmups=args.warmups,
        samples=args.samples,
        status=args.status,
        label=args.label,
    )
    print(json.dumps(payload["summary"], indent=2))
    if not payload["summary"]["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
