# SwiftSci 3.13.2 Deep Performance Audit

> **Navigation**: [← v3.13.2 Evidence Index](README.md) · [Performance Overview](../../../PERFORMANCE/3.13.2.md) · [Deep Accuracy Audit](accuracy.deep.md) · [Accuracy Overview](../../../ACCURACY/3.13.2.md)

Release: [v3.13.2](https://github.com/Nodibell/SwiftSci/releases/tag/v3.13.2) · [Source commit](https://github.com/Nodibell/SwiftSci/commit/8039d02f9d63c27022a794dfff5bd43d2e8a9710)

- Release commit: `8039d02f9d63c27022a794dfff5bd43d2e8a9710` (clean)
- Source tree SHA-256: `953e773629eabf731530b82edd6d74813e8d5f4b22bfec25b412ba9951eb4a9c`
- Benchmark worker SHA-256: `535b15de13233d243d3f76ef0072cf1d309ce47ab5db5ae0b61894d001abd734`
- Scientific result SHA-256: `9c3f79b7ac7afdd833de6a130237ff60a035946dc3807b1da2e397d7d7f241eb`
- [Release evidence index](README.md)

Apple M4 Pro (arm64), 24 GiB RAM, macOS 27.2.1 build 26B131, AC power. Xcode 27.2 (27B52), Swift 6.4.2. The Release worker was verified free of coverage instrumentation and built with Metal safe math and precise FP32 functions.

Standardized runs used Python 3.14.6 with pandas 3.0.6, NumPy 2.5.3, SciPy 1.18.1, PyArrow 25.0.1, Polars 1.44.2 and DuckDB 1.5.6 where applicable. Each process used two warmups and five measured samples.

All six certificates passed (4,810 validated measurements). The archive includes reports and raw run files for each campaign.

| Campaign | Cases | Engines | Policy | Process batches | Validated samples |
|---|---:|---|---|---:|---:|
| standard | 14 | pandas, swiftsci | legacy-capped | 3 | 420 |
| standard-production-default | 14 | pandas, swiftsci | production-default | 5 | 700 |
| nist | 27 | pandas, swiftsci | legacy-capped | 3 | 810 |
| public-data | 11 | duckdb, pandas, polars, swiftsci | legacy-capped | 3 | 660 |
| migration | 36 | pandas, swiftsci | legacy-capped | 3 | 1080 |
| tabular-migration | 19 | duckdb, pandas, polars, swiftsci | legacy-capped | 3 | 1140 |

## Workloads and input sizes

The standard profiles cover eager CSV loading, filtering, sorting, grouping, feature extraction, scaling and descriptive statistics. Migration adds mixed-type tables, joins, streaming CSV, Parquet and broader numerical operations. Public-data workloads use H2O-style grouping fixtures and normalized UCI Wine Quality red data. NIST covers mean, sample variance and sample standard deviation across nine reference datasets.

| Dataset | Rows | Input bytes |
|---|---:|---:|
| `h2o-medium-hot-key` | 100,000 | 3,277,220 |
| `h2o-medium-uniform` | 100,000 | 3,867,483 |
| `large` | 1,000,000 | 23,804,730 |
| `medium` | 100,000 | 2,181,600 |
| `mixed-medium` | 100,000 | 3,440,517 |
| `nist-lew` | 200 | 3,518 |
| `nist-lottery` | 218 | 3,750 |
| `nist-mavro` | 50 | 2,187 |
| `nist-michelso` | 100 | 2,349 |
| `nist-numacc1` | 3 | 1,840 |
| `nist-numacc2` | 1,001 | 11,794 |
| `nist-numacc3` | 1,001 | 14,830 |
| `nist-numacc4` | 1,001 | 15,765 |
| `nist-pidigits` | 5,000 | 46,441 |
| `parquet-medium` | 100,000 | 637,791 |
| `tiny` | 129 | 2,372 |
| `wine-quality-red` | 1,599 | 91,063 |

The [dataset manifests](../../Specs/datasets/) provide generator settings or upstream references and input hashes. Scientific inputs include seeded series and model fixtures and the AirPassengers series. Their exact settings, horizons and split sizes are in the [scientific suite](../../Tools/scientific_validation_suite.py).

## Timing boundaries and variability

The worker measures wall time through materialized output. CSV parsing is inside the timed read. Fixture preparation, process startup, imports, warmup and reference validation are outside measured samples. The standardized summaries report the median and sample standard deviation across process medians. The scientific summaries report the interquartile range (IQR) across process medians for fit and inference separately. Times below are in milliseconds.

## Standard profile

Times are medians of process medians in milliseconds. The ratio is pandas time divided by SwiftSci time. Dashes indicate timing below 1 µs in at least one engine. Peak RSS in the raw reports is process-wide.

| Profile | Case | SwiftSci median ± SD (ms) | pandas median ± SD (ms) | pandas / SwiftSci |
|---|---|---:|---:|---:|
| standard | `medium-csv-read` | 2.267500 ± 0.113537 | 9.686041 ± 0.096907 | 4.27× |
| standard | `medium-filter` | 0.225250 ± 0.005892 | 0.695792 ± 0.096900 | 3.09× |
| standard | `medium-sort` | 2.266542 ± 0.046140 | 2.705834 ± 0.113343 | 1.19× |
| standard | `medium-group-sum` | 0.384041 ± 0.007032 | 0.504333 ± 0.002923 | 1.31× |
| standard | `medium-flat-matrix` | 0.108084 ± 0.002167 | 0.482458 ± 0.111062 | 4.46× |
| standard | `medium-target` | 0.051916 ± 0.002519 | 0.044917 ± 0.002002 | 0.87× |
| standard | `medium-standard-scale` | 2.006209 ± 0.046156 | 0.610875 ± 0.168703 | 0.30× |
| standard | `medium-minmax-scale` | 0.769084 ± 0.003473 | 0.546750 ± 0.033148 | 0.71× |
| standard | `medium-mean` | 0.007084 ± 0.000487 | 0.015000 ± 0.000312 | 2.12× |
| standard | `medium-variance` | 0.052583 ± 0.000105 | 0.055667 ± 0.000512 | 1.06× |
| standard | `medium-stddev` | 0.052416 ± 0.002632 | 0.055750 ± 0.001038 | 1.06× |
| standard | `nist-mean` | 0.000208 ± 0.000024 | 0.003333 ± 0.000209 | — |
| standard | `nist-variance` | 0.000709 ± 0.000024 | 0.007250 ± 0.000230 | — |
| standard | `nist-stddev` | 0.000750 ± 0.000042 | 0.007750 ± 0.000409 | — |
| standard-production-default | `medium-csv-read` | 2.203792 ± 0.465107 | 10.451250 ± 0.392783 | 4.74× |
| standard-production-default | `medium-filter` | 0.224667 ± 0.017085 | 0.587208 ± 0.158002 | 2.61× |
| standard-production-default | `medium-sort` | 2.150375 ± 0.076679 | 2.431416 ± 0.097298 | 1.13× |
| standard-production-default | `medium-group-sum` | 0.378333 ± 0.004153 | 0.509500 ± 0.005988 | 1.35× |
| standard-production-default | `medium-flat-matrix` | 0.109291 ± 0.002991 | 0.450916 ± 0.029610 | 4.13× |
| standard-production-default | `medium-target` | 0.050458 ± 0.000090 | 0.030583 ± 0.003938 | 0.61× |
| standard-production-default | `medium-standard-scale` | 1.960708 ± 0.028622 | 0.624459 ± 0.139823 | 0.32× |
| standard-production-default | `medium-minmax-scale` | 0.753917 ± 0.015647 | 0.503541 ± 0.097877 | 0.67× |
| standard-production-default | `medium-mean` | 0.007500 ± 0.000369 | 0.014917 ± 0.000652 | 1.99× |
| standard-production-default | `medium-variance` | 0.052542 ± 0.003165 | 0.055500 ± 0.048796 | 1.06× |
| standard-production-default | `medium-stddev` | 0.052625 ± 0.030944 | 0.057584 ± 0.000581 | 1.09× |
| standard-production-default | `nist-mean` | 0.000208 ± 0.000018 | 0.003292 ± 0.000205 | — |
| standard-production-default | `nist-variance` | 0.000750 ± 0.000029 | 0.007375 ± 0.000109 | — |
| standard-production-default | `nist-stddev` | 0.000750 ± 0.000023 | 0.007708 ± 0.000201 | — |

A ratio below 1 favors pandas. It leads in target vector extraction, standard scaling and min–max scaling in both standard policies. Scientific examples where Python leads include OLS and LinearSVC fitting. In an unmatched diagnostic, the 500-point optimized Holt–Winters fit measured 0.262875 ms (SwiftSci IQR 0.008125 ms) and 47.156625 ms (statsmodels IQR 0.415625 ms). The libraries optimize different quantities, so this case yields no comparable speedup.

## Scientific comparisons

Five independent processes per workload with two warmups and five measured samples each. The Python reference used Python 3.11.9, NumPy 2.3.5, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0, statsmodels 0.14.6 and SHAP 0.51.0. Cross-engine ratios below are limited to comparisons cleared by the scientific accuracy and comparability checks. Four stages in the full suite were timing-resolution limited.

| Workload | Stage | SwiftSci median (IQR), ms | Python median (IQR), ms | Faster engine | Factor |
|---|---|---:|---:|---|---:|
| `ts_hw_syn500_fixed` | fit | 0.010250 (0.000375) | 1.798208 (0.037208) | SwiftSci | 175.43× |
| `ts_hw_syn50k_fixed` | fit | 1.144292 (0.003666) | 137.005125 (0.244458) | SwiftSci | 119.73× |
| `ts_hw_syn50k_fixed` | infer | 0.047291 (0.000541) | 135.839666 (0.790166) | SwiftSci | 2872.42× |
| `ts_arima_syn500_matched_hr` | fit | 0.039208 (0.000458) | 3.237375 (0.044667) | SwiftSci | 82.57× |
| `ts_arima_syn500_matched_hr` | infer | 0.001292 (0.000041) | 0.336833 (0.009708) | SwiftSci | 260.71× |
| `ts_arima_dgp_syn50k_matched_hr` | fit | 2.420500 (0.113417) | 207.323542 (2.337626) | SwiftSci | 85.65× |
| `ts_arima_dgp_syn50k_matched_hr` | infer | 0.079125 (0.003417) | 0.379292 (0.035125) | SwiftSci | 4.79× |
| `ts_arima_dgp_syn500_matched_hr` | fit | 0.038709 (0.000499) | 3.300666 (0.023959) | SwiftSci | 85.27× |
| `ts_arima_dgp_syn500_matched_hr` | infer | 0.001334 (0.000250) | 0.341000 (0.001624) | SwiftSci | 255.62× |
| `ml_ols_regression` | fit | 0.439667 (0.001707) | 0.256791 (0.003333) | Python | 1.71× |
| `ml_ols_regression` | infer | 0.001500 (0.000041) | 0.023875 (0.004417) | SwiftSci | 15.92× |
| `ml_linear_svc` | fit | 12.134250 (0.302666) | 0.610167 (0.018916) | Python | 19.89× |
| `ml_linear_svc` | infer | 0.002083 (0.000042) | 0.057834 (0.008458) | SwiftSci | 27.76× |
| `ml_decision_tree_regressor` | fit | 1.779959 (0.017083) | 3.343292 (0.028875) | SwiftSci | 1.88× |
| `ml_decision_tree_regressor` | infer | 0.004000 (0.000167) | 0.025459 (0.001917) | SwiftSci | 6.36× |
| `ml_decision_tree_classifier` | fit | 3.726459 (0.026917) | 3.747875 (0.029292) | SwiftSci | 1.01× |
| `ml_decision_tree_classifier` | infer | 0.012833 (0.000333) | 0.071750 (0.001375) | SwiftSci | 5.59× |
| `ml_random_forest_regressor` | fit | 10.232292 (0.171041) | 128.860750 (0.253209) | SwiftSci | 12.59× |
| `ml_random_forest_regressor` | infer | 0.216250 (0.001667) | 1.784417 (0.024084) | SwiftSci | 8.25× |
| `ml_random_forest_classifier` | fit | 15.524958 (0.104750) | 58.435500 (0.542249) | SwiftSci | 3.76× |
| `ml_random_forest_classifier` | infer | 1.066542 (0.157750) | 3.669791 (0.029833) | SwiftSci | 3.44× |
| `ml_gbdt_regressor` | fit | 20.629625 (0.533457) | 115.045709 (1.018583) | SwiftSci | 5.58× |
| `ml_gbdt_regressor` | infer | 0.055333 (0.001083) | 0.201125 (0.004541) | SwiftSci | 3.63× |
| `ml_hist_gbdt_regressor` | fit | 12.845458 (0.027166) | 18.109458 (0.113792) | SwiftSci | 1.41× |
| `ml_hist_gbdt_regressor` | infer | 0.201875 (0.002333) | 0.405625 (0.008750) | SwiftSci | 2.01× |
| `ml_hist_gbdt_classifier` | fit | 14.678791 (0.086916) | 23.797833 (0.121125) | SwiftSci | 1.62× |
| `ml_hist_gbdt_classifier` | infer | 0.482333 (0.006291) | 1.236333 (0.029332) | SwiftSci | 2.56× |
| `cluster_pca` | fit | 0.144917 (0.001875) | 0.244791 (0.005625) | SwiftSci | 1.69× |
| `cluster_pca` | infer | 0.070792 (0.001083) | 0.038709 (0.004667) | Python | 1.83× |
| `cluster_kmeans` | fit | 0.208334 (0.003291) | 0.519125 (0.017834) | SwiftSci | 2.49× |
| `cluster_kmeans` | infer | 0.039417 (0.000501) | 0.062458 (0.004166) | SwiftSci | 1.58× |
| `stats_ddof1_100k` | fit | 1.128166 (0.016417) | 1.163625 (0.051834) | SwiftSci | 1.03× |
| `stats_ddof1_100k` | infer | 1.128166 (0.016417) | 1.163625 (0.051834) | SwiftSci | 1.03× |

### SwiftLLM KV-cache ablation

These rows compare two SwiftSci paths on the same synthetic workload.

| Workload | Full forward (ms) | KV cache (ms) | Full forward / KV cache |
|---|---:|---:|---:|
| `llm_kvcache_ablation_p24_d16` | 6.376917 | 5.715458 | 1.12× |
| `llm_kvcache_ablation_p64_d24` | 9.707750 | 8.586709 | 1.13× |

## Interpretation and reproduction

Use the same input, output contract, estimator and execution policy. Legacy-capped and production-default use different thread settings. Actual active thread counts were not measured. Ratios are suppressed for unresolved timing and unmatched scientific stages. Peak RSS includes libraries, runtime and warmup state. These results do not establish universal RAM savings or a cross-version speedup.

Start at the [tagged source](https://github.com/Nodibell/SwiftSci/tree/v3.13.2), follow the [runner guide](../../README.md), and use the [resolved plans and raw results](README.md). See the [deep accuracy audit](accuracy.deep.md) for acceptance and comparator scope.
