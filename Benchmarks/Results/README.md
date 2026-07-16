# Benchmark results

Files in this directory are generated at runtime and should **not** be committed
(except this README).

## Generate & compare

```bash
# from SwiftAnalytics/SwiftAnalytics (package root)
mkdir -p Benchmarks/Results

swift run -c release SwiftAnalyticsBenchmarks \
  --json Benchmarks/Results/swift_results.json

python3 Benchmarks/Python/benchmarks.py \
  --json Benchmarks/Results/python_results.json

python3 Benchmarks/Python/compare.py \
  Benchmarks/Results/swift_results.json \
  Benchmarks/Results/python_results.json
```

## CI gate vs informational

`compare.py` prints every matched pair. Only a small **CI gate** set can fail
the process (exit 1) when Swift is >2× slower than Python:

| Gated (must stay competitive) | Informational (tracked for 0.8) |
| ----------------------------- | ------------------------------- |
| Pearson correlation           | Mean / StdDev / Variance        |
| Holt-Winters, ARIMA fit/forecast | CSV / Filter / GroupBy / Sort |
| Kalman Filter 1D              | RandomForest / GBDT / KMeans / PCA |
| TS Decomposition additive     | LinearRegression (near-parity after SGD align) |

Use `--gate-all` to apply the threshold to every matched pair (local audits).

## Fairness notes (v0.7)

- **LinearRegression** — Python uses `SGDRegressor` (100 epochs), matching
  Swift’s gradient-descent fit (not closed-form OLS).
- **Kalman Filter 1D** — Python runs a NumPy constant-velocity filter aligned
  with `KalmanFilter.oneDimensional` (not an EWM stand-in).
- **TS Decomposition** — Python uses `statsmodels.tsa.seasonal.seasonal_decompose`.
