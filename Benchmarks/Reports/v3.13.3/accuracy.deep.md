# SwiftSci 3.13.3 Deep Accuracy & Conformance Audit

> **Navigation**: [← v3.13.3 Evidence Index](README.md) · [Accuracy Overview](../../../ACCURACY/3.13.3.md) · [Deep Performance Audit](performance.deep.md) · [Performance Overview](../../../PERFORMANCE/3.13.3.md)

Branch: `release/3.13.3` · Base commit: `dd3a8d980c81bb2e6f15ce4fb4400bd1bb959320` (`dirty: true` with `3.13.3` changes)

- Source tree SHA-256 (`summary.json` / `suite.json`): `334c6d57af4ceec2d8f4c1ca8e54f9d3f62bab2e817b56d4d791294639171b04`
- Benchmark worker SHA-256 (on disk): `a6c568e291bc9358fd1b7633347b71840858be7245ff0548dd9ebe2af1221776`
- Benchmark worker SHA-256 (recorded in `scientific_validation.json` / `pretrained-inspection/controls.json`): `c907ae3c5f6b8b82d0de7e6047391d8bc109084bc4a06179ec3a51a3320461a2`
- Scientific validation JSON SHA-256: `0184a50235c818f9f5daa5e516e6d83891210644b6d259251f706e9011cff62d`
- Release-gate `acceptance.json` SHA-256: `b59ee8a1d2aa81e46e77cb125df20402034c240f29b0c76c4c2b4e90b8ce8112`
- Release-gate `summary.json` SHA-256: `8aa5845f8e97dc572489833652ff9d767d5c6d2ed788cbae2eb0b3b65b197a3a`
- Comparison `suite.json` SHA-256: `dc94c9a2757a3b41255a2c97d3e025dfe028c8e35aa7763f686ba753f38a4643`
- Comparison `cpu-acceptance/acceptance.json` SHA-256: `043a6f6ff59f26e7ab65435d27e7e7f5648234f1359834bd175bef4b21360035`
- [Release evidence index](README.md)

## Release validation

- **Scientific suite**: `39/39` workloads passed, `3/3` rolling-origin backtests passed, `2/2` expected-error checks verified (`all_gates_passed: true`, AC power).
- **Standardized comparison (`v3.13.3-comparison`, AC `production-default`)**: `6/6` profiles passed (`standard`, `extended`, `public-data`, `nist`, `migration`, `tabular-migration`), `10,300` validated measurements across `412` engine-case rows, `0` failures.
- **CPU acceptance (`16` profiles)**: `14/16` profiles passed strictly (`400/404` total engine-case executions) and recorded `4` known **pandas** Wampler 4/5 comparator mismatches in `numerical-binary64` and `numerical-conformance`. SwiftSci passed `202/202` engine-case executions; `regression-policy.json` recorded `blocking: []`.
- **Auxiliary controls**: `60/60` pretrained inspection controls (`pretrained-inspection/controls.json`) and `100/100` workflow mutation controls (`workflow-controls/controls.json`) passed.
- **Swift & DocC verification**: `xcodebuild test` + `swift test -c release` passed `1,874` test cases (`0` failures; `7` env-gated skips). `verify_doc_coverage.py` reported `100.00% (2,551/2,551 matched declarations documented)`. `CookbookVerification` passed `71/71` recipes.

## 16-Profile CPU Acceptance Summary

Both `v3.13.3-release-gate/acceptance/acceptance.json` and `v3.13.3-comparison/cpu-acceptance/acceptance.json` produced identical profile counts (`400/404` passed):

| Profile | Engine Cases | Passed | Failures | Status | Run SHA-256 (release-gate) |
|---|---:|---:|---:|---|---|
| `vision-conformance` | 16 | 16 | 0 | `passed` | `d1c6d95b00fb30b3045276d38262104df8a4dbff1815b610b29ee3f911e121b6` |
| `boundary-cpu-conformance` | 8 | 8 | 0 | `passed` | `7ed378e4aa6ece3f88a54568ae6b0b9cb7bb546ac501095ea988bde47768d82c` |
| `certification` | 54 | 54 | 0 | `passed` | `a2aff6337b3b59e73f688281ad6ee3759448757905c234870f2b7283cb3e4592` |
| `controlled-conformance` | 20 | 20 | 0 | `passed` | `0e516797688867c46678485d78a333122ac74968bf160ab34e821641eee19d91` |
| `dataframe-conformance` | 48 | 48 | 0 | `passed` | `1b9d97298eb9383c55d26cfeb070a565052597bc246bd82571ba1fe2f1946997` |
| `migration-smoke` | 72 | 72 | 0 | `passed` | `959154959fb79bdccb4d26d5b399e11f7479be5ebc20fd2bc8b6eb5927502e5d` |
| `model-conformance` | 6 | 6 | 0 | `passed` | `539d803c9a3e881ed7da372137900d39e00d4deab8047d8a2501f402743a0244` |
| `neural-cpu-conformance` | 10 | 10 | 0 | `passed` | `8b0ac36f54167cf324782acceb406239ccf858aa752d99ab113a95ff4f824f4f` |
| `numerical-binary64` | 38 | 36 | 2 (pandas) | `failed` | `0fbafc2810106137632789ef57e566813790cfe7d08ccb0015de1d527356903c` |
| `numerical-conformance` | 38 | 36 | 2 (pandas) | `failed` | `1d0d4f377dfece2602635c11888bce12e5cd1e43a5e78739756aa3e4ece75506` |
| `public-smoke` | 22 | 22 | 0 | `passed` | `fa8bf4c9a707b96d19407575266abd7c8c2dbe61a9a07274dfa529d352d2ff58` |
| `smoke` | 28 | 28 | 0 | `passed` | `bbeaab2c97c507d58401de104020d7dce86722960d862039b794641c5807b6d3` |
| `supervised-conformance` | 12 | 12 | 0 | `passed` | `94f94e707913985fb6a66758b5ab6a96906788b8c315703e3a3b6105efae70a6` |
| `scientific-workflow` | 8 | 8 | 0 | `passed` | `f0fd916e22115273f386d6d2047b40b57c4fca2bea2165a9f1d14c3b8f0238a8` |
| `persisted-workflow` | 12 | 12 | 0 | `passed` | `c98c0e8d376b58ac2fd23a69d0b0c08eda7be9be6c65c4268e8057812a43a600` |
| `neural-shaped-cpu-conformance` | 12 | 12 | 0 | `passed` | `7baf1d1f30c174e9f78c2c9a8b9c9a258c3e79be12ccadb712f0b8526189b142` |

## Reference methods and tolerances

Each standardized contract declares shape, output ordering, units and tolerances. For finite numeric values the harness applies `abs(actual - expected) <= atol + rtol * abs(expected)`; NaN and infinity are checked separately where declared by the reference. Tabular workflows also check null semantics and round trips, including independent Parquet readback. Scientific checks use NumPy/SciPy, matched Python reference methods, scikit-learn and statsmodels where applicable, and independently enumerated coalitions for selected TreeSHAP cases.

### NIST univariate cases (`v3.13.3-comparison/measurements.json`)

All `27` mean, sample variance and sample standard deviation cases passed against NIST Statistical Reference Dataset values.

| NIST dataset | Rows | Gates passed | Mean abs error | Variance abs error | StdDev abs error | Max absolute SwiftSci error |
|---|---:|---:|---:|---:|---:|---:|
| `nist-lew` | 200 | 3/3 | `0` | `7.27596e-11` | `1.13687e-13` | `7.27596e-11` |
| `nist-lottery` | 218 | 3/3 | `3.41061e-13` | `4.36557e-11` | `5.68434e-14` | `4.36557e-11` |
| `nist-mavro` | 50 | 3/3 | `0` | `2.78197e-20` | `3.24176e-17` | `3.24176e-17` |
| `nist-michelso` | 100 | 3/3 | `0` | `1.79544e-16` | `1.13798e-15` | `1.13798e-15` |
| `nist-numacc1` | 3 | 3/3 | `0` | `0` | `0` | `0` |
| `nist-numacc2` | 1,001 | 3/3 | `0` | `1.73472e-18` | `1.38778e-17` | `1.38778e-17` |
| `nist-numacc3` | 1,001 | 3/3 | `3.49246e-10` | `6.98492e-12` | `3.49246e-11` | `3.49246e-10` |
| `nist-numacc4` | 1,001 | 3/3 | `5.58794e-09` | `1.11759e-10` | `5.58794e-10` | `5.58794e-09` |
| `nist-pidigits` | 5,000 | 3/3 | `0` | `1.06581e-14` | `1.77636e-15` | `1.06581e-14` |

## 3.13.3 Numerical & Correctness Gates

| Check | Result | Scope |
|---|---|---|
| `MLP` central finite-difference gradient check (`ε = 1e-6`) | Max error `< 1e-6` | ReLU, Sigmoid, Tanh across binary BCE, 3-class & 5-class softmax, and MSE; full batch (`0..<8`) and partial final batch (`8..<11`) |
| `MLP` extreme logit stability | Exact finite loss & gradients | `b = [1000, -1000, 999]` (`1 + log1p(exp(-1))` within `1e-12`), `b = [-1000, -1000]` (`log(2)`, `gradB = [-0.5, 0.5]` within `1e-12`), BCE `b = [1000]` (`500.0` within `1e-9`) |
| `MLPClassifier` 768-D (`N=512, D=768, C=5, batchSize=32, maxIter=20`) | `train_accuracy = 1.0000` | Matches scikit-learn `1.0000` training accuracy with `40.312 ms` vs `103.639 ms` fit |
| OLS (`ml_ols_regression`) | Max prediction difference `5.50671e-14` | Matched seeded fixture (`r2 = 0.998018`) |
| Logistic regression (`ml_logistic_regression`) | Holdout accuracy `0.9600` (SwiftSci) vs `0.9575` (scikit-learn) | One fixed split |
| LinearSVC (`ml_linear_svc`) | Holdout accuracy `0.9575` (SwiftSci) vs `0.9575` (scikit-learn) | Exact `0.0` accuracy gap |
| PCA (`cluster_pca`) | Total explained-variance ratio difference `4.44089e-16` | Seeded projection fixture (`1500 × 4`) |
| K-Means (`cluster_kmeans`) | WCSS relative gap `5.57234e-16` | `6528.643055` vs `6528.643055` |
| Decision tree TreeSHAP (`explain_treeshap_dt_k4`) | Coalition-oracle diff `0`, additivity error `1.77636e-15` | `25` instances, `1` tree |
| Random forest TreeSHAP (`explain_treeshap_rf_k8`) | Coalition-oracle diff `4.44089e-16`, additivity error `4.44089e-16` | `25` instances, `10` trees |
| Sample moments (`stats_ddof1_100k`) | Skewness diff `2.38524e-17`, excess kurtosis diff `2.13163e-14` | `100,000` values vs SciPy translated to `ddof=1` |
| G-037 one-step fitted values | Max warmup diff `0.0`; max residual identity diff `5.68434e-14` | Verified across `ts_arima_*` and `ts_sarima_*` workloads |
| LLM KV-cache ablation | Max abs logit diff `5.96046e-07` (`p24_d16`) / `4.17233e-07` (`p64_d24`) | Synthetic Transformer forward vs KV-cache equivalence |

## Rolling-origin mean RMSE

| Backtest | SwiftSci | Python reference | Seasonal naive |
|---|---:|---:|---:|
| `hw_syn500_ro` | `1.511027` | `1.554041` | `5.347851` |
| `arima_dgp_ro` | `5.223510` | `5.071532` | `4.276752` |
| `sarima_ap_ro` | `32.542239` | `30.992804` | `38.991669` |

## Scientific workload outcomes (39/39 passed)

| Workload | Group | Gate | Key Metric / Notes |
|---|---|---|---|
| `ts_hw_syn500_opt` | `SwiftForecast` | Passed | RMSE `0.750040` (Swift) vs `0.712150` (Python) |
| `ts_hw_syn500_fixed` | `SwiftForecast` | Passed | RMSE `1.547220` (Swift) vs `1.627154` (Python) |
| `ts_hw_syn50k_fixed` | `SwiftForecast` | Passed | RMSE `1.420978` (Swift) vs `1.435845` (Python) |
| `ts_arima_syn500_matched_hr` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `1.42109e-14` |
| `ts_arima_syn500_vs_statespace` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `1.42109e-14` |
| `ts_arima_syn50k_vs_statespace` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `5.68434e-14` |
| `ts_arima_dgp_syn50k_matched_hr` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `5.68434e-14` |
| `ts_arima_dgp_syn500_matched_hr` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `1.42109e-14` |
| `ts_sarima_expect_fail_syn500` | `SwiftForecast` | Expected error verified | Rejects invalid/non-invertible polynomial |
| `ts_sarima_expect_fail_syn50k` | `SwiftForecast` | Expected error verified | Rejects invalid/non-invertible polynomial |
| `ts_sarima_airpassengers_110_010` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `5.68434e-14` |
| `ts_sarima_airpassengers_011_011` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `5.68434e-14` |
| `ts_sarima_airpassengers_144_g037` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `5.68434e-14` |
| `ts_auto_arima_valid` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `0.0` |
| `ts_auto_arima_zero_variance` | `SwiftForecast` | Passed | Constant-series guard verified |
| `ts_auto_arima_poly_sign_check` | `SwiftForecast` | Passed | Schur–Cohn sign alignment verified |
| `ts_auto_arima_poly_unstable_check` | `SwiftForecast` | Passed | Unstable polynomial rejection verified |
| `anomaly_syn144_s04` | `SwiftForecast` | Passed | SHESD seasonal anomaly detection |
| `anomaly_syn500_holdout` | `SwiftForecast` | Passed | SHESD holdout anomaly detection |
| `anomaly_syn500_trended_diagnostic` | `SwiftForecast` | Passed | SHESD trended diagnostic |
| `ml_ols_regression` | `SwiftML` | Passed | Max pred diff `5.50671e-14` |
| `ml_logistic_regression` | `SwiftML` | Passed | Accuracy `0.9600` vs `0.9575` |
| `ml_linear_svc` | `SwiftML` | Passed | Accuracy `0.9575` vs `0.9575` (`gap = 0.0`) |
| `ml_decision_tree_regressor` | `SwiftML` | Passed | R² `0.600367` vs `0.601527` |
| `ml_decision_tree_classifier` | `SwiftML` | Passed | Accuracy `0.7450` vs `0.7450` (`gap = 0.0`) |
| `ml_random_forest_regressor` | `SwiftML` | Passed | R² `0.604400` vs `0.662001` |
| `ml_random_forest_classifier` | `SwiftML` | Passed | Accuracy `0.8600` vs `0.8500` |
| `ml_gbdt_regressor` | `SwiftML` | Passed | R² `0.728881` vs `0.731315` |
| `ml_hist_gbdt_regressor` | `SwiftML` | Passed | R² `0.786131` vs `0.777853` |
| `ml_hist_gbdt_classifier` | `SwiftML` | Passed | Accuracy `0.8625` vs `0.8700` |
| `cluster_pca` | `SwiftCluster` | Passed | EVR abs diff `4.44089e-16` |
| `cluster_kmeans` | `SwiftCluster` | Passed | WCSS rel gap `5.57234e-16` |
| `explain_treeshap_dt_k4` | `SwiftExplain` | Passed | Coalition-oracle diff `0.0`, additivity `1.77636e-15` |
| `explain_treeshap_rf_k8` | `SwiftExplain` | Passed | Coalition-oracle diff `4.44089e-16`, additivity `4.44089e-16` |
| `explain_treeshap_dt_k16_mc` | `SwiftExplain` | Passed | MC (`2,048` permutations) additivity `7.37188e-14` |
| `stats_ddof1_small5` | `SwiftStats` | Passed | Skew diff `0.0`, kurt diff `4.44089e-16` |
| `stats_ddof1_100k` | `SwiftStats` | Passed | Skew diff `2.38524e-17`, kurt diff `2.13163e-14` |
| `llm_kvcache_ablation_p24_d16` | `SwiftLLM` | Passed | Max abs logit diff `5.96046e-07` |
| `llm_kvcache_ablation_p64_d24` | `SwiftLLM` | Passed | Max abs logit diff `4.17233e-07` |

## Comparator failures and deferred coverage

| CPU profile | pandas case | Reported mismatch |
|---|---|---|
| `numerical-binary64` | `nist-wampler4-binary64` | `Output mismatch at 0: 1.0000000028854616 != 1.0` |
| `numerical-binary64` | `nist-wampler5-binary64` | `Output mismatch at 0: 1.0000003429656807 != 1.0` |
| `numerical-conformance` | `nist-wampler4-decimal` | `Output mismatch at 0: 1.0000000028854616 != 1.0` |
| `numerical-conformance` | `nist-wampler5-decimal` | `Output mismatch at 0: 1.0000003429656807 != 1.0` |

- **G-035**: `TreeSHAP` for `GradientBoostedTreesRegressor` remains unsupported.
- **G-036**: Class-targeted probability `TreeSHAP` remains deferred (the classifier adapter explains the encoded class-label output).
- **Environment-gated Swift tests**: `7` environment-gated Swift tests (external HuggingFace checkpoint / live network / optional fixture checks) were skipped across `xcodebuild test` and `swift test -c release`.

See the [evidence index](README.md) for gate thresholds and cryptographic hashes, and the [deep performance audit](performance.deep.md) for full 6-profile comparison tables.
