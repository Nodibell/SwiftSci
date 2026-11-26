# SwiftSci 3.13.2 Deep Accuracy & Conformance Audit

> **Navigation**: [← v3.13.2 Evidence Index](README.md) · [Accuracy Overview](../../../ACCURACY/3.13.2.md) · [Deep Performance Audit](performance.deep.md) · [Performance Overview](../../../PERFORMANCE/3.13.2.md)

Release: [v3.13.2](https://github.com/Nodibell/SwiftSci/releases/tag/v3.13.2) · [Source commit](https://github.com/Nodibell/SwiftSci/commit/8039d02f9d63c27022a794dfff5bd43d2e8a9710)

- Release commit: `8039d02f9d63c27022a794dfff5bd43d2e8a9710` (clean)
- Source tree SHA-256: `953e773629eabf731530b82edd6d74813e8d5f4b22bfec25b412ba9951eb4a9c`
- Benchmark worker SHA-256: `535b15de13233d243d3f76ef0072cf1d309ce47ab5db5ae0b61894d001abd734`
- Scientific result SHA-256: `9c3f79b7ac7afdd833de6a130237ff60a035946dc3807b1da2e397d7d7f241eb`
- [Release evidence index](README.md)

## Release validation

- Scientific suite: 39/39 workloads, 3/3 rolling-origin backtests, and both expected-error checks.
- G-037: 4 Swift regression tests, 9 scientific workloads and 6 rolling-origin folds passed. Maximum warmup difference: 0. Maximum residual identity difference: 5.68434e-14. ARIMA/SARIMA reconstruct one-step fitted values using observed levels, and ARIMAX includes exogenous terms.
- Standardized conformance: 6/6 internal workload certificates and 4,810 validated measurements.
- CPU acceptance: all 16 profiles ran. Strict conformance passed 14 profiles and recorded four pandas Wampler failures in two profiles. SwiftSci passed 202/202 engine-case executions. The reviewed regression policy had zero blockers.
- Auxiliary controls: 60 pretrained inspection and 40 workflow mutation controls passed.
- Swift Release builds and tests completed, including Core ML. Python passed 240/240 tests including MLX GPU checks. DocC generated 14/14 archives and the declaration scan reported 2,551/2,551.

## Reference methods and tolerances

Each standardized contract declares shape, output ordering, units and tolerances. For finite numeric values the harness applies `abs(actual - expected) <= atol + rtol * abs(expected)`; NaN and infinity are checked separately where declared by the reference. Tabular workflows also check null semantics and round trips, including independent Parquet readback. [Workload definitions](../../Specs/workloads/) and the archived resolved plans give per-case tolerances. Scientific checks use NumPy/SciPy, matched Python reference methods, scikit-learn and statsmodels where applicable, and independently enumerated coalitions for selected TreeSHAP cases.

### NIST univariate cases

All 27 mean, sample variance and sample standard deviation cases passed against NIST Statistical Reference Dataset values. The maximum below spans quantities with different scales. Compare each result with its own tolerance. For PiDigits mean, for example, absolute tolerance is 3.552713678800501e-15 and relative tolerance is 1e-14.

| NIST dataset | Rows | Gates passed | Max absolute SwiftSci error across outputs |
|---|---:|---:|---:|
| `nist-lew` | 200 | 3/3 | 7.27596e-11 |
| `nist-lottery` | 218 | 3/3 | 4.36557e-11 |
| `nist-mavro` | 50 | 3/3 | 3.24176e-17 |
| `nist-michelso` | 100 | 3/3 | 1.13798e-15 |
| `nist-numacc1` | 3 | 3/3 | 0 |
| `nist-numacc2` | 1,001 | 3/3 | 1.38778e-17 |
| `nist-numacc3` | 1,001 | 3/3 | 3.49246e-10 |
| `nist-numacc4` | 1,001 | 3/3 | 5.58794e-09 |
| `nist-pidigits` | 5,000 | 3/3 | 1.06581e-14 |

## Numerical safeguards and fitted-value correctness

ARIMA and SARIMA guard against non-finite intermediates and invalid MA/SMA polynomials. Failed fits clear fitted state. AutoARIMA applies the corrected Schur–Cohn coefficient sign convention to candidate orders.

For observation index `t`, warmup is `d` for ARIMA and `d + D*s` for SARIMA. Warmup fitted entries retain observations. Beyond that boundary, `residuals[t - warmUp] = series[t] - fittedValues[t]` within floating-point precision. Reconstructing one-step fitted values uses observed levels, and ARIMAX adds the exogenous contribution. G-037 verifies this contract. It does not establish out-of-sample accuracy.

| Example | Result | Scope |
|---|---|---|
| OLS | Maximum prediction difference 5.50671e-14 | Matched seeded fixture |
| Logistic regression | Holdout accuracy 0.9600 (SwiftSci) vs 0.9575 (reference) | One fixed split |
| PCA | Total explained-variance ratio difference 4.44089e-16 | Seeded projection fixture |
| Four-feature decision tree TreeSHAP | Coalition-oracle difference 0, additivity error 1.77636e-15 | Same fixture and leaf weights |
| Sample moments (100,000 values) | Skewness difference 2.71051e-17, excess kurtosis difference 2.13163e-14 | SciPy reference translated to the SwiftSci sample denominator |

## Rolling-origin mean RMSE

| Backtest | SwiftSci | Python reference | Seasonal naive |
|---|---:|---:|---:|
| `hw_syn500_ro` | 1.511027 | 1.554041 | 5.347851 |
| `arima_dgp_ro` | 5.223510 | 5.071532 | 4.276752 |
| `sarima_ap_ro` | 32.542239 | 30.992804 | 38.991669 |

Each row aggregates three chronological cutoffs. The seasonal-naive baseline had lower mean RMSE than either fitted model in the ARIMA data-generator case. Passing an implementation check does not guarantee a predictive improvement.

## Scientific workload outcomes

| Workload | Gate |
|---|---|
| `ts_hw_syn500_opt` | Passed |
| `ts_hw_syn500_fixed` | Passed |
| `ts_hw_syn50k_fixed` | Passed |
| `ts_arima_syn500_matched_hr` | Passed |
| `ts_arima_syn500_vs_statespace` | Passed |
| `ts_arima_syn50k_vs_statespace` | Passed |
| `ts_arima_dgp_syn50k_matched_hr` | Passed |
| `ts_arima_dgp_syn500_matched_hr` | Passed |
| `ts_sarima_expect_fail_syn500` | Expected error verified |
| `ts_sarima_expect_fail_syn50k` | Expected error verified |
| `ts_sarima_airpassengers_110_010` | Passed |
| `ts_sarima_airpassengers_011_011` | Passed |
| `ts_sarima_airpassengers_144_g037` | Passed |
| `ts_auto_arima_valid` | Passed |
| `ts_auto_arima_zero_variance` | Passed |
| `ts_auto_arima_poly_sign_check` | Passed |
| `ts_auto_arima_poly_unstable_check` | Passed |
| `anomaly_syn144_s04` | Passed |
| `anomaly_syn500_holdout` | Passed |
| `anomaly_syn500_trended_diagnostic` | Passed |
| `ml_ols_regression` | Passed |
| `ml_logistic_regression` | Passed |
| `ml_linear_svc` | Passed |
| `ml_decision_tree_regressor` | Passed |
| `ml_decision_tree_classifier` | Passed |
| `ml_random_forest_regressor` | Passed |
| `ml_random_forest_classifier` | Passed |
| `ml_gbdt_regressor` | Passed |
| `ml_hist_gbdt_regressor` | Passed |
| `ml_hist_gbdt_classifier` | Passed |
| `cluster_pca` | Passed |
| `cluster_kmeans` | Passed |
| `explain_treeshap_dt_k4` | Passed |
| `explain_treeshap_rf_k8` | Passed |
| `explain_treeshap_dt_k16_mc` | Passed |
| `stats_ddof1_small5` | Passed |
| `stats_ddof1_100k` | Passed |
| `llm_kvcache_ablation_p24_d16` | Passed |
| `llm_kvcache_ablation_p64_d24` | Passed |

## Comparator failures and deferred coverage

| CPU profile | pandas case | Reported mismatch |
|---|---|---|
| `numerical-binary64` | `nist-wampler4-binary64` | Output mismatch at 0: 1.0000000028854616 != 1.0 |
| `numerical-binary64` | `nist-wampler5-binary64` | Output mismatch at 0: 1.0000003429656807 != 1.0 |
| `numerical-conformance` | `nist-wampler4-decimal` | Output mismatch at 0: 1.0000000028854616 != 1.0 |
| `numerical-conformance` | `nist-wampler5-decimal` | Output mismatch at 0: 1.0000003429656807 != 1.0 |

The conditional Swift Llama parity checks below were skipped:

- Checkpoint chat IDs match without a duplicate BOS
- Checkpoint IDs, BOS processing and byte-exact decoding match the pinned reference
- Live HuggingFace MLX checkpoint parity against tokenizers & mlx_lm

- G-035: TreeSHAP for gradient-boosted tree regressors remains unsupported.
- G-036: class-targeted probability TreeSHAP remains deferred. The classifier adapter explains the encoded class-label output.

See the [evidence index](README.md) for gate thresholds, comparator outputs and acceptance decisions, and the [deep performance audit](performance.deep.md) for timing boundaries.
