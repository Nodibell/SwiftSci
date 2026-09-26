#!/usr/bin/env python3
"""
compare.py — Benchmark Methodology v2 Comparison Engine.

Categorizes and compares Swift and Python benchmark results into:
  1. Strict Apple-to-Apple (Identical Datasets, Algorithms & Parameters)
  2. Library-to-Library (Ecosystem Parity: SwiftSci vs Scikit-Learn/Statsmodels)
  3. Architecture & Hardware Specialization (Metal GPU, SIMD MSL, KV-Cache)

Usage:
    python3 compare.py swift_results.json python_results.json
    python3 compare.py swift_results.json python_results.json --markdown
    python3 compare.py swift_results.json python_results.json --gate-all
"""

import argparse
import json
import re
import sys
from typing import Dict, List, Optional, Tuple


# Category definitions
CAT_APPLE_TO_APPLE = "1. Strict Apple-to-Apple (C/SIMD Parity & Identical Fixtures)"
CAT_LIBRARY_TO_LIBRARY = "2. Library-to-Library (Ecosystem Parity: SwiftSci vs Scikit-Learn / Statsmodels)"
CAT_HARDWARE_SPECIALIZED = "3. Architecture & Hardware Specialization (Metal GPU, SIMD MSL, KV-Cache)"
CAT_OTHER = "4. Other / Exploratory"

# Categorization mapping by normalized key prefix or substring
CATEGORIES = {
    # ── Category 1: Strict Apple-to-Apple ──
    "mean": CAT_APPLE_TO_APPLE,
    "stddev": CAT_APPLE_TO_APPLE,
    "variance": CAT_APPLE_TO_APPLE,
    "pearson correlation": CAT_APPLE_TO_APPLE,
    "two-sample t-test": CAT_APPLE_TO_APPLE,
    "spearman correlation": CAT_APPLE_TO_APPLE,
    "csv read": CAT_APPLE_TO_APPLE,
    "csv stream read": CAT_APPLE_TO_APPLE,
    "csv stream + filter": CAT_APPLE_TO_APPLE,
    "csv stream + groupby": CAT_APPLE_TO_APPLE,
    "filter rows": CAT_APPLE_TO_APPLE,
    "groupby + sum/mean": CAT_APPLE_TO_APPLE,
    "sortby double column": CAT_APPLE_TO_APPLE,
    "dataframe simd hash join": CAT_APPLE_TO_APPLE,
    "forecast errors suite": CAT_APPLE_TO_APPLE,
    "classification roc-auc": CAT_APPLE_TO_APPLE,
    "onehotencoder fittransform": CAT_APPLE_TO_APPLE,

    # ── Category 2: Library-to-Library ──
    "holt-winters fit": CAT_LIBRARY_TO_LIBRARY,
    "arima fit": CAT_LIBRARY_TO_LIBRARY,
    "arima forecast": CAT_LIBRARY_TO_LIBRARY,
    "randomforest fit": CAT_LIBRARY_TO_LIBRARY,
    "gbdt regressor fit": CAT_LIBRARY_TO_LIBRARY,
    "kmeans fit": CAT_LIBRARY_TO_LIBRARY,
    "pca svd fittransform": CAT_LIBRARY_TO_LIBRARY,
    "pca svd fit": CAT_LIBRARY_TO_LIBRARY,
    "naivebayesclassifier fit": CAT_LIBRARY_TO_LIBRARY,
    "isolationforest fit": CAT_LIBRARY_TO_LIBRARY,
    "kernelshap explain": CAT_LIBRARY_TO_LIBRARY,
    "treeshap explanation": CAT_LIBRARY_TO_LIBRARY,
    "linearregression fit": CAT_LIBRARY_TO_LIBRARY,

    # ── Category 3: Architecture & Specialization ──
    "linearsvc fit": CAT_HARDWARE_SPECIALIZED,
    "vectorstore cosine search": CAT_HARDWARE_SPECIALIZED,
    "global average pooling": CAT_HARDWARE_SPECIALIZED,
    "sqlite direct dataframe ingestion": CAT_HARDWARE_SPECIALIZED,
    "metal": CAT_HARDWARE_SPECIALIZED,
    "kv-cache": CAT_HARDWARE_SPECIALIZED,
    "rope": CAT_HARDWARE_SPECIALIZED,
    "ringlwe": CAT_HARDWARE_SPECIALIZED,
    "pnns": CAT_HARDWARE_SPECIALIZED,
}

# Gated benchmarks in CI regression testing
CI_GATE_KEYS = frozenset({
    "pearson correlation",
    "holt-winters fit",
    "arima fit",
    "arima forecast horizon=24",
    "kalman filter 1d",
    "randomforest fit",
    "gbdt regressor fit",
    "kernelshap explain",
    "ringlwe encrypt/decrypt",
    "pnns classify",
})


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def normalize(name: str) -> str:
    """Strip parenthetical qualifiers like '(NumPy, …)' / '(1k pts, …)' and synonyms."""
    cleaned = re.sub(r"\s*\([^)]*\)\s*", " ", name).strip().lower()
    cleaned = cleaned.replace("rank correlation", "correlation")
    return cleaned


def categorize(key: str) -> str:
    for pattern, cat in CATEGORIES.items():
        if pattern in key:
            return cat
    return CAT_OTHER


def format_relative(speedup: Optional[float]) -> str:
    if speedup is None:
        return "n/a"
    if speedup >= 1.0:
        return f"SwiftSci {speedup:.2f}×"
    else:
        return f"Python {1.0 / speedup:.2f}×"


def print_table_text(category_title: str, rows: List[dict]):
    header = (
        f"{'Benchmark':<50}  {'Swift(ms)':>10}  {'Python(ms)':>10}  "
        f"{'Relative':>16}  {'Gate':>6}"
    )
    print(f"\n📂 {category_title}")
    print("═" * len(header))
    print(header)
    print("─" * len(header))

    for r in rows:
        gate_tag = "CI" if r["gated"] else "info"
        if r["python_ms"] is None:
            print(f"  {r['name']:<48}  {r['swift_ms']:10.3f}  {'n/a':>10}  "
                  f"{'n/a':>16}  {gate_tag:>6}")
            continue

        rel_str = format_relative(r["speedup"])
        print(f"  {r['name']:<48}  {r['swift_ms']:10.3f}  {r['python_ms']:10.3f}  "
              f"{rel_str:>16}  {gate_tag:>6}")
    print("═" * len(header))


def print_table_markdown(category_title: str, rows: List[dict]):
    print(f"\n### {category_title}\n")
    print("| Benchmark | SwiftSci (ms) | Python (ms) | Relative Performance | Status |")
    print("|:---|---:|---:|:---:|:---:|")
    for r in rows:
        swift_str = f"{r['swift_ms']:.3f}"
        python_str = f"{r['python_ms']:.3f}" if r['python_ms'] is not None else "n/a"
        rel_str = format_relative(r["speedup"])
        gate_tag = "Gated" if r["gated"] else "Informational"
        print(f"| {r['name']} | {swift_str} | {python_str} | **{rel_str}** | {gate_tag} |")


def main():
    parser = argparse.ArgumentParser(description="Compare Swift vs Python benchmark results (v2 Methodology)")
    parser.add_argument("swift_json", help="Path to swift_results.json")
    parser.add_argument("python_json", help="Path to python_results.json")
    parser.add_argument(
        "--regression-threshold", type=float, default=1.2,
        help="Speedup ratio below which a CI failure is triggered",
    )
    parser.add_argument(
        "--min-ms-threshold", type=float, default=1.0,
        help="Minimum Swift execution time to trigger a regression failure",
    )
    parser.add_argument(
        "--gate-all", action="store_true",
        help="Apply regression threshold to every matched pair",
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="Output comparison as formatted GitHub Flavored Markdown",
    )
    args = parser.parse_args()

    swift_report = load(args.swift_json)
    python_report = load(args.python_json)

    python_index = {normalize(r["name"]): r for r in python_report["results"]}

    categorized_rows: Dict[str, List[dict]] = {
        CAT_APPLE_TO_APPLE: [],
        CAT_LIBRARY_TO_LIBRARY: [],
        CAT_HARDWARE_SPECIALIZED: [],
        CAT_OTHER: [],
    }

    regression_failures = []
    known_gaps = []

    for sr in swift_report["results"]:
        key = normalize(sr["name"])
        pr = python_index.get(key)
        if pr is None:
            candidates = [k for k in python_index if k in key or key in k]
            pr = python_index[candidates[0]] if candidates else None
            if pr is not None:
                key = normalize(pr["name"])

        gated = args.gate_all or any(gk == key or key.startswith(gk) for gk in CI_GATE_KEYS)
        cat = categorize(key)

        if pr:
            speedup = pr["medianMs"] / sr["medianMs"] if sr["medianMs"] > 0 else float("inf")
            row = {
                "name": sr["name"],
                "key": key,
                "category": cat,
                "swift_ms": sr["medianMs"],
                "python_ms": pr["medianMs"],
                "speedup": speedup,
                "faster": speedup >= 1.0,
                "gated": gated,
            }
            categorized_rows[cat].append(row)

            if speedup < (1.0 / args.regression_threshold):
                if sr["medianMs"] < args.min_ms_threshold:
                    known_gaps.append(row)
                elif gated:
                    regression_failures.append(row)
                else:
                    known_gaps.append(row)
        else:
            row = {
                "name": sr["name"],
                "key": key,
                "category": cat,
                "swift_ms": sr["medianMs"],
                "python_ms": None,
                "speedup": None,
                "faster": None,
                "gated": gated,
            }
            categorized_rows[cat].append(row)

    # Render report
    for cat_name in [CAT_APPLE_TO_APPLE, CAT_LIBRARY_TO_LIBRARY, CAT_HARDWARE_SPECIALIZED, CAT_OTHER]:
        rows = categorized_rows[cat_name]
        if not rows:
            continue
        if args.markdown:
            print_table_markdown(cat_name, rows)
        else:
            print_table_text(cat_name, rows)

    # Global summary
    all_matched = [r for sub in categorized_rows.values() for r in sub if r["python_ms"] is not None]
    swift_wins = sum(1 for r in all_matched if r["faster"])
    python_wins = sum(1 for r in all_matched if not r["faster"])

    print(f"\n📊 Summary: SwiftSci faster: {swift_wins} | Python faster: {python_wins}")

    if regression_failures:
        print(f"\n⚠️  REGRESSION: {len(regression_failures)} gated benchmark(s) "
              f"where Swift is >{args.regression_threshold:.1f}× slower than Python:")
        for r in regression_failures:
            print(f"    • {r['name']} — Swift {r['swift_ms']:.3f} ms vs "
                  f"Python {r['python_ms']:.3f} ms (speedup {r['speedup']:.2f}×)")
        print("\n❌ CI CHECK FAILED — performance regression detected.")
        sys.exit(1)

    print("\n✅ CI CHECK PASSED — no regressions detected across verified suites.")
    sys.exit(0)


if __name__ == "__main__":
    main()
