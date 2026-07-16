#!/usr/bin/env python3
"""
compare.py — Compare Swift and Python benchmark results.

Usage:
    python3 compare.py swift_results.json python_results.json

Prints a full comparison table. Only benchmarks in CI_GATE are allowed to
fail the process (exit 1) when Swift is slower than --regression-threshold.
Known gaps (DataFrame, trees, etc.) are reported as informational until 0.8.
"""

import argparse
import json
import re
import sys


# Normalized name keys that participate in the CI regression gate.
# Keep this list to pairs that are algorithmically comparable and where
# Swift is expected to stay competitive (Forecast + Pearson).
# Everything else is printed but does not fail CI (tracked for 0.8).
CI_GATE_KEYS = frozenset({
    "pearson correlation",
    "holt-winters fit",
    "arima fit",
    "arima forecast horizon=24",
    "kalman filter 1d",
    "ts decomposition additive",
})


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def normalize(name: str) -> str:
    """Strip parenthetical qualifiers like '(NumPy, …)' / '(1k pts, …)'."""
    return re.sub(r"\s*\([^)]*\)\s*", " ", name).strip().lower()


def main():
    parser = argparse.ArgumentParser(description="Compare Swift vs Python benchmark results")
    parser.add_argument("swift_json",  help="Path to swift_results.json")
    parser.add_argument("python_json", help="Path to python_results.json")
    parser.add_argument(
        "--regression-threshold", type=float, default=2.0,
        help="Speedup ratio below which a CI failure is triggered "
             "(default: 2.0 = Swift must not be >2× slower on gated benches)",
    )
    parser.add_argument(
        "--gate-all", action="store_true",
        help="Apply the regression threshold to every matched pair (ignore CI_GATE)",
    )
    args = parser.parse_args()

    swift_report  = load(args.swift_json)
    python_report = load(args.python_json)

    python_index = {normalize(r["name"]): r for r in python_report["results"]}

    rows = []
    for sr in swift_report["results"]:
        key = normalize(sr["name"])
        pr = python_index.get(key)
        if pr is None:
            candidates = [k for k in python_index if k in key or key in k]
            pr = python_index[candidates[0]] if candidates else None
            if pr is not None:
                key = normalize(pr["name"])

        gated = args.gate_all or key in CI_GATE_KEYS

        if pr:
            speedup = pr["medianMs"] / sr["medianMs"] if sr["medianMs"] > 0 else float("inf")
            rows.append({
                "name":      sr["name"],
                "key":       key,
                "swift_ms":  sr["medianMs"],
                "python_ms": pr["medianMs"],
                "speedup":   speedup,
                "faster":    speedup >= 1.0,
                "gated":     gated,
            })
        else:
            rows.append({
                "name":      sr["name"],
                "key":       key,
                "swift_ms":  sr["medianMs"],
                "python_ms": None,
                "speedup":   None,
                "faster":    None,
                "gated":     gated,
            })

    header = (
        f"{'Benchmark':<52}  {'Swift(ms)':>10}  {'Python(ms)':>10}  "
        f"{'Speedup':>9}  {'Winner':>8}  {'Gate':>6}"
    )
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))

    regression_failures = []
    known_gaps = []

    for r in rows:
        gate_tag = "CI" if r["gated"] else "info"
        if r["python_ms"] is None:
            print(f"  {r['name']:<50}  {r['swift_ms']:10.3f}  {'n/a':>10}  "
                  f"{'n/a':>9}  {'?':>8}  {gate_tag:>6}")
            continue

        speedup = r["speedup"]
        winner  = "🟢 Swift" if speedup >= 1.0 else "🔴 Python"
        sp_str  = f"{speedup:.2f}×"
        print(f"  {r['name']:<50}  {r['swift_ms']:10.3f}  {r['python_ms']:10.3f}  "
              f"{sp_str:>9}  {winner:>8}  {gate_tag:>6}")

        if speedup < (1.0 / args.regression_threshold):
            if r["gated"]:
                regression_failures.append(r)
            else:
                known_gaps.append(r)

    print("═" * len(header))

    swift_wins  = sum(1 for r in rows if r.get("faster"))
    python_wins = sum(1 for r in rows if r.get("faster") is False)
    print(f"\n  🟢 Swift faster: {swift_wins} benchmarks")
    print(f"  🔴 Python faster: {python_wins} benchmarks")

    if known_gaps:
        print(f"\n  ℹ️  Known gaps (informational, not gated — tracked for 0.8):")
        for r in known_gaps:
            print(f"    • {r['name']} — Swift {r['swift_ms']:.3f} ms vs "
                  f"Python {r['python_ms']:.3f} ms (speedup {r['speedup']:.2f}×)")

    if regression_failures:
        print(f"\n  ⚠️  REGRESSION: {len(regression_failures)} gated benchmark(s) "
              f"where Swift is >{args.regression_threshold:.1f}× slower than Python:")
        for r in regression_failures:
            print(f"    • {r['name']} — Swift {r['swift_ms']:.3f} ms vs "
                  f"Python {r['python_ms']:.3f} ms (speedup {r['speedup']:.2f}×)")
        print("\n  ❌ CI CHECK FAILED — performance regression detected.")
        sys.exit(1)

    print("\n  ✅ CI CHECK PASSED — no gated regressions detected.")
    sys.exit(0)


if __name__ == "__main__":
    main()
