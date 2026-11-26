# SwiftSci 3.13.3 Deep Performance Audit

> **Navigation**: [← v3.13.3 Evidence Index](README.md) · [Performance Overview](../../../PERFORMANCE/3.13.3.md) · [Deep Accuracy Audit](accuracy.deep.md) · [Accuracy Overview](../../../ACCURACY/3.13.3.md)

Branch: `release/3.13.3` · Base commit: `dd3a8d980c81bb2e6f15ce4fb4400bd1bb959320` (`dirty: true` with `3.13.3` changes)

- Source tree SHA-256 (`summary.json` / `suite.json`): `334c6d57af4ceec2d8f4c1ca8e54f9d3f62bab2e817b56d4d791294639171b04`
- Benchmark worker SHA-256 (on disk): `a6c568e291bc9358fd1b7633347b71840858be7245ff0548dd9ebe2af1221776`
- Benchmark worker SHA-256 (recorded in `scientific_validation.json` / `pretrained-inspection/controls.json`): `c907ae3c5f6b8b82d0de7e6047391d8bc109084bc4a06179ec3a51a3320461a2`
- Scientific validation JSON SHA-256: `0184a50235c818f9f5daa5e516e6d83891210644b6d259251f706e9011cff62d`
- Release-gate `acceptance.json` SHA-256: `b59ee8a1d2aa81e46e77cb125df20402034c240f29b0c76c4c2b4e90b8ce8112`
- Release-gate `summary.json` SHA-256: `8aa5845f8e97dc572489833652ff9d767d5c6d2ed788cbae2eb0b3b65b197a3a`
- Comparison `suite.json` SHA-256: `dc94c9a2757a3b41255a2c97d3e025dfe028c8e35aa7763f686ba753f38a4643`
- Comparison `measurements.json` SHA-256: `f1d57e17c99916044094f277f6bca40f4d6b1cd905fbdf9d25746fb6b6847237`
- Comparison `cpu-acceptance/acceptance.json` SHA-256: `043a6f6ff59f26e7ab65435d27e7e7f5648234f1359834bd175bef4b21360035`
- [Release evidence index](README.md)

Apple M4 Pro (`arm64`), 24 GiB RAM (`25,769,803,776` bytes), `macOS 27.2.1` build `26B131`, AC power (`power_start = AC`, `power_end = AC`, `power_unchanged = true` across all 6 `v3.13.3-comparison` profiles and `scientific_validation.json`). Xcode `27.2 (27B52)`, Swift `6.4.2`. The Release worker was verified free of coverage instrumentation and built with `MTL_FAST_MATH=NO`, `MTL_MATH_MODE=SAFE`.

All standardized and scientific runs used Python `3.14.6` (`numpy 2.5.3`, `pandas 3.0.6`, `scipy 1.18.1`, `pyarrow 25.0.1`, `polars 1.44.2`, `duckdb 1.5.6`, `scikit-learn 1.9.1`, `statsmodels 0.15.0`, `shap 0.54.1`). Each standardized worker process executed 2 warmups and 5 measured samples across 5 rotating fresh processes (`25` samples per engine-case).

All six `production-default` certificates passed (`10,300` validated measurements across `412` engine-case rows, `0` failures).

| Campaign | Cases | Engines | Policy | Process rounds | Validated samples |
|---|---:|---|---|---:|---:|
| `standard` | 14 | duckdb, pandas, polars, swiftsci | `production-default` (AC) | 5 | 1,400 |
| `extended` | 14 | duckdb, pandas, polars, swiftsci | `production-default` (AC) | 5 | 1,400 |
| `public-data` | 11 | duckdb, pandas, polars, swiftsci | `production-default` (AC) | 5 | 1,100 |
| `nist` | 27 | duckdb, pandas, polars, swiftsci | `production-default` (AC) | 5 | 2,700 |
| `migration` | 36 | pandas, swiftsci | `production-default` (AC) | 5 | 1,800 |
| `tabular-migration` | 19 | duckdb, pandas, polars, swiftsci | `production-default` (AC) | 5 | 1,900 |

## Workloads and input sizes

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

## Timing boundaries and variability

The worker measures wall time through materialized output. CSV parsing is inside the timed read; DuckDB fetches results into Arrow tables or NumPy arrays inside timing. Fixture preparation, process startup, imports, warmup, and reference validation are outside measured samples. Standardized comparison tables report median and sample standard deviation (`median ± SD`) across 5 process medians in milliseconds; scientific summaries report `median (IQR)` in milliseconds. Winner columns always state the faster engine first (`SwiftSci X.XX×`, `Pandas X.XX×`, `Polars X.XX×`, `DuckDB X.XX×`, `Python X.XX×`), and dashes (`—`) mark unresolved `< 1 µs` timings.

## Standard profile (`standard`, 100k rows)

| Case | SwiftSci (ms) | Pandas (ms) | Polars (ms) | DuckDB (ms) | SwiftSci vs Pandas | SwiftSci vs Polars | SwiftSci vs DuckDB |
|---|---:|---:|---:|---:|---|---|---|
| `medium-csv-read` | 2.403458 ± 0.082164 | 10.305166 ± 0.122066 | 1.294042 ± 0.157819 | 25.780958 ± 0.056505 | SwiftSci 4.29× | Polars 1.86× | SwiftSci 10.73× |
| `medium-filter` | 0.229792 ± 0.003150 | 0.725125 ± 0.011870 | 0.673042 ± 0.067326 | 0.994459 ± 0.039708 | SwiftSci 3.16× | SwiftSci 2.93× | SwiftSci 4.33× |
| `medium-sort` | 2.275292 ± 0.062081 | 2.779125 ± 0.077118 | 1.369458 ± 0.219040 | 5.990375 ± 0.028353 | SwiftSci 1.22× | Polars 1.66× | SwiftSci 2.63× |
| `medium-group-sum` | 0.391458 ± 0.101899 | 0.541042 ± 0.010228 | 0.622625 ± 0.151686 | 0.334584 ± 0.054920 | SwiftSci 1.38× | SwiftSci 1.59× | DuckDB 1.17× |
| `medium-flat-matrix` | 0.114667 ± 0.003415 | 0.704291 ± 0.055790 | 1.185791 ± 0.051811 | 1.480750 ± 0.076819 | SwiftSci 6.14× | SwiftSci 10.34× | SwiftSci 12.91× |
| `medium-target` | 0.058917 ± 0.002773 | 0.056500 ± 0.020674 | 0.072542 ± 0.012716 | 0.999708 ± 0.093740 | Pandas 1.04× | SwiftSci 1.23× | SwiftSci 16.97× |
| `medium-standard-scale` | 2.017834 ± 0.043865 | 0.894542 ± 0.036536 | 1.213417 ± 0.013436 | 2.490125 ± 0.084260 | Pandas 2.26× | Polars 1.66× | SwiftSci 1.23× |
| `medium-minmax-scale` | 0.782625 ± 0.005658 | 0.801667 ± 0.032417 | 1.365959 ± 0.058062 | 2.743167 ± 0.035069 | SwiftSci 1.02× | SwiftSci 1.75× | SwiftSci 3.51× |
| `medium-mean` | 0.007959 ± 0.000349 | 0.015167 ± 0.000225 | 0.027625 ± 0.000353 | 0.194250 ± 0.010582 | SwiftSci 1.91× | SwiftSci 3.47× | SwiftSci 24.41× |
| `medium-variance` | 0.055375 ± 0.001186 | 0.055458 ± 0.000479 | 0.062083 ± 0.001457 | 0.513042 ± 0.027287 | SwiftSci 1.00× | SwiftSci 1.12× | SwiftSci 9.26× |
| `medium-stddev` | 0.052750 ± 0.001915 | 0.055667 ± 0.000737 | 0.061125 ± 0.000940 | 0.515458 ± 0.023612 | SwiftSci 1.06× | SwiftSci 1.16× | SwiftSci 9.77× |
| `nist-mean` | unresolved (0.000208) | 0.003416 ± 0.000267 | 0.013000 ± 0.000363 | 0.102125 ± 0.015140 | — | — | — |
| `nist-variance` | unresolved (0.000750) | 0.007250 ± 0.000414 | 0.013709 ± 0.000294 | 0.118667 ± 0.013667 | — | — | — |
| `nist-stddev` | unresolved (0.000750) | 0.008000 ± 0.000231 | 0.013666 ± 0.000621 | 0.112416 ± 0.018042 | — | — | — |

## Extended profile (`extended`, 1M rows)

| Case | SwiftSci (ms) | Pandas (ms) | Polars (ms) | DuckDB (ms) | SwiftSci vs Pandas | SwiftSci vs Polars | SwiftSci vs DuckDB |
|---|---:|---:|---:|---:|---|---|---|
| `large-csv-read` | 23.521166 ± 0.751921 | 94.882250 ± 1.420425 | 6.115167 ± 0.615166 | 53.730625 ± 0.089807 | SwiftSci 4.03× | Polars 3.85× | SwiftSci 2.28× |
| `large-filter` | 2.176750 ± 0.032524 | 4.725750 ± 0.081537 | 1.129250 ± 0.042435 | 6.015250 ± 0.123867 | SwiftSci 2.17× | Polars 1.93× | SwiftSci 2.76× |
| `large-sort` | 23.136541 ± 0.972311 | 31.616792 ± 0.513400 | 10.800750 ± 0.862740 | 20.351792 ± 0.271983 | SwiftSci 1.37× | Polars 2.14× | DuckDB 1.14× |
| `large-group-sum` | 3.828042 ± 0.080660 | 3.781917 ± 0.056335 | 2.848000 ± 0.138263 | 0.590416 ± 0.017099 | Pandas 1.01× | Polars 1.34× | DuckDB 6.48× |
| `large-flat-matrix` | 1.142333 ± 0.025769 | 2.758167 ± 0.014041 | 2.090000 ± 0.219107 | 6.135708 ± 0.355744 | SwiftSci 2.41× | SwiftSci 1.83× | SwiftSci 5.37× |
| `large-target` | 0.580833 ± 0.083017 | 0.237750 ± 0.007962 | 0.613500 ± 0.047703 | 4.131333 ± 0.307243 | Pandas 2.44× | SwiftSci 1.06× | SwiftSci 7.11× |
| `large-standard-scale` | 20.011291 ± 0.288727 | 4.637333 ± 0.031446 | 3.641625 ± 0.255810 | 10.551209 ± 0.300905 | Pandas 4.32× | Polars 5.50× | DuckDB 1.90× |
| `large-minmax-scale` | 9.580125 ± 0.983297 | 3.751791 ± 0.055401 | 4.183250 ± 0.215222 | 10.428834 ± 0.342565 | Pandas 2.55× | Polars 2.29× | SwiftSci 1.09× |
| `large-mean` | 0.079458 ± 0.001188 | 0.118541 ± 0.001886 | 0.104417 ± 0.002922 | 0.295208 ± 0.023141 | SwiftSci 1.49× | SwiftSci 1.31× | SwiftSci 3.72× |
| `large-variance` | 0.524250 ± 0.002421 | 0.479750 ± 0.003814 | 0.458125 ± 0.015371 | 0.708542 ± 0.091053 | Pandas 1.09× | Polars 1.14× | SwiftSci 1.35× |
| `large-stddev` | 0.524625 ± 0.001125 | 0.481041 ± 0.001028 | 0.467709 ± 0.009694 | 0.748125 ± 0.082113 | Pandas 1.09× | Polars 1.12× | SwiftSci 1.43× |
| `nist-mean` | unresolved (0.000208) | 0.003333 ± 0.000183 | 0.013333 ± 0.000868 | 0.104000 ± 0.008334 | — | — | — |
| `nist-variance` | unresolved (0.000792) | 0.007416 ± 0.000151 | 0.013708 ± 0.000120 | 0.134000 ± 0.010429 | — | — | — |
| `nist-stddev` | unresolved (0.000792) | 0.007666 ± 0.000441 | 0.013584 ± 0.000708 | 0.122709 ± 0.005276 | — | — | — |

## Public-data profile (`public-data`)

| Case | SwiftSci (ms) | Pandas (ms) | Polars (ms) | DuckDB (ms) | SwiftSci vs Pandas | SwiftSci vs Polars | SwiftSci vs DuckDB |
|---|---:|---:|---:|---:|---|---|---|
| `medium-uniform-h2o-q1` | 1.928958 ± 0.085813 | 1.223083 ± 0.038694 | 1.132750 ± 0.069885 | 1.428125 ± 0.030478 | Pandas 1.58× | Polars 1.70× | DuckDB 1.35× |
| `medium-uniform-h2o-q2` | 6.832708 ± 0.058531 | 2.878917 ± 0.169704 | 1.870250 ± 0.200916 | 2.508292 ± 0.091264 | Pandas 2.37× | Polars 3.65× | DuckDB 2.72× |
| `medium-uniform-h2o-q3` | 2.529875 ± 0.025332 | 1.810125 ± 0.071734 | 1.335000 ± 0.031750 | 1.579291 ± 0.032861 | Pandas 1.40× | Polars 1.90× | DuckDB 1.60× |
| `medium-uniform-h2o-q4` | 0.655667 ± 0.117688 | 1.078125 ± 0.010098 | 1.071958 ± 0.060342 | 0.510125 ± 0.063236 | SwiftSci 1.64× | SwiftSci 1.63× | DuckDB 1.29× |
| `medium-uniform-h2o-q5` | 0.856958 ± 0.022571 | 1.078833 ± 0.090672 | 1.087208 ± 0.030660 | 0.430167 ± 0.043787 | SwiftSci 1.26× | SwiftSci 1.27× | DuckDB 1.99× |
| `medium-hot-key-h2o-q1` | 1.869875 ± 0.055934 | 1.160084 ± 0.031521 | 1.178375 ± 0.073422 | 1.539417 ± 0.041941 | Pandas 1.61× | Polars 1.59× | DuckDB 1.21× |
| `medium-hot-key-h2o-q2` | 5.645042 ± 0.124532 | 2.413958 ± 0.097850 | 2.047042 ± 0.171870 | 2.511959 ± 0.036531 | Pandas 2.34× | Polars 2.76× | DuckDB 2.25× |
| `medium-hot-key-h2o-q3` | 2.354333 ± 0.030644 | 1.852375 ± 0.045195 | 1.866333 ± 0.109073 | 1.936041 ± 0.033627 | Pandas 1.27× | Polars 1.26× | DuckDB 1.22× |
| `medium-hot-key-h2o-q4` | 1.199292 ± 0.024209 | 2.225500 ± 0.034423 | 1.172625 ± 0.030992 | 1.014750 ± 0.023718 | SwiftSci 1.86× | Polars 1.02× | DuckDB 1.18× |
| `medium-hot-key-h2o-q5` | 1.376125 ± 0.021346 | 1.719458 ± 0.091099 | 1.378125 ± 0.052379 | 0.992791 ± 0.033581 | SwiftSci 1.25× | SwiftSci 1.00× | DuckDB 1.39× |
| `wine-red-pipeline` | 0.385458 ± 0.024197 | 1.480459 ± 0.146229 | 1.136250 ± 0.068849 | 18.179500 ± 0.176310 | SwiftSci 3.84× | SwiftSci 2.95× | SwiftSci 47.16× |

## Tabular-migration profile (`tabular-migration`)

| Case | SwiftSci (ms) | Pandas (ms) | Polars (ms) | DuckDB (ms) | SwiftSci vs Pandas | SwiftSci vs Polars | SwiftSci vs DuckDB |
|---|---:|---:|---:|---:|---|---|---|
| `medium-group-sum-mean` | 0.515834 ± 0.013769 | 1.697791 ± 0.034668 | 0.660375 ± 0.033129 | 0.371708 ± 0.037655 | SwiftSci 3.29× | SwiftSci 1.28× | DuckDB 1.39× |
| `medium-row-sum` | 3.913542 ± 0.108713 | 19.097292 ± 0.160556 | 23.926750 ± 0.202131 | 17.275042 ± 0.122751 | SwiftSci 4.88× | SwiftSci 6.11× | SwiftSci 4.41× |
| `medium-inner-join` | 2.058084 ± 0.036578 | 0.837166 ± 0.034202 | 1.731625 ± 0.163117 | 2.197833 ± 0.047308 | Pandas 2.46× | Polars 1.19× | SwiftSci 1.07× |
| `medium-pearson` | 0.162083 ± 0.010408 | 0.250292 ± 0.007042 | 0.145375 ± 0.005902 | 0.510708 ± 0.013024 | SwiftSci 1.54× | Polars 1.11× | SwiftSci 3.15× |
| `medium-spearman` | 3.016958 ± 0.030355 | 3.371708 ± 0.094850 | 2.792959 ± 0.047782 | 10.288417 ± 0.113834 | SwiftSci 1.12× | Polars 1.08× | SwiftSci 3.41× |
| `mixed-medium-csv-read` | 8.077250 ± 0.463623 | 13.936209 ± 0.160318 | 1.624292 ± 0.086236 | 36.018042 ± 0.160174 | SwiftSci 1.73× | Polars 4.97× | SwiftSci 4.46× |
| `mixed-medium-filter` | 0.286416 ± 0.009053 | 0.827459 ± 0.051253 | 0.704250 ± 0.070308 | 1.423250 ± 0.030628 | SwiftSci 2.89× | SwiftSci 2.46× | SwiftSci 4.97× |
| `mixed-medium-group-sum-mean` | 2.239458 ± 0.244793 | 2.115875 ± 0.084607 | 0.825334 ± 0.073892 | 1.273916 ± 0.060033 | Pandas 1.06× | Polars 2.71× | DuckDB 1.76× |
| `mixed-medium-sort` | 2.360834 ± 0.025103 | 3.181000 ± 0.043555 | 1.671500 ± 0.146803 | 7.158625 ± 0.069987 | SwiftSci 1.35× | Polars 1.41× | SwiftSci 3.03× |
| `mixed-medium-flat-matrix` | 0.107375 ± 0.002807 | 0.631292 ± 0.060889 | 1.207042 ± 0.141507 | 1.436625 ± 0.063654 | SwiftSci 5.88× | SwiftSci 11.24× | SwiftSci 13.38× |
| `mixed-medium-row-sum` | 3.868375 ± 0.105269 | 47.490792 ± 0.355454 | 27.183750 ± 0.448637 | 24.106750 ± 0.228839 | SwiftSci 12.28× | SwiftSci 7.03× | SwiftSci 6.23× |
| `mixed-medium-inner-join` | 2.079334 ± 0.014743 | 0.899542 ± 0.047857 | 2.093916 ± 0.220973 | 3.026042 ± 0.050991 | Pandas 2.31× | SwiftSci 1.01× | SwiftSci 1.46× |
| `mixed-medium-parquet-write` | 7.876833 ± 0.075884 | 7.834709 ± 0.130399 | 1.668334 ± 0.061884 | 4.024208 ± 0.063219 | Pandas 1.01× | Polars 4.72× | DuckDB 1.96× |
| `mixed-medium-standard-scale` | 2.004375 ± 0.032860 | 0.893000 ± 0.063191 | 1.511667 ± 0.131597 | 2.454834 ± 0.063862 | Pandas 2.24× | Polars 1.33× | SwiftSci 1.22× |
| `mixed-medium-minmax-scale` | 0.772209 ± 0.010456 | 0.756416 ± 0.047203 | 1.625875 ± 0.081853 | 2.613000 ± 0.102079 | Pandas 1.02× | SwiftSci 2.11× | SwiftSci 3.38× |
| `parquet-medium-read` | 23.288500 ± 0.313420 | 1.847000 ± 0.067771 | 1.778667 ± 0.041135 | 2.789750 ± 0.120115 | Pandas 12.61× | Polars 13.09× | DuckDB 8.35× |
| `large-mean` | 0.078583 ± 0.001288 | 0.119958 ± 0.005048 | 0.106000 ± 0.019037 | 0.295917 ± 0.018463 | SwiftSci 1.53× | SwiftSci 1.35× | SwiftSci 3.77× |
| `large-variance` | 0.521541 ± 0.006330 | 0.482125 ± 0.006326 | 0.454167 ± 0.006961 | 0.713875 ± 0.075564 | Pandas 1.08× | Polars 1.15× | SwiftSci 1.37× |
| `large-stddev` | 0.527750 ± 0.006199 | 0.490000 ± 0.007550 | 0.451542 ± 0.007801 | 0.702750 ± 0.019044 | Pandas 1.08× | Polars 1.17× | SwiftSci 1.33× |

## Migration profile (`migration`, 36 cases)

| Case | SwiftSci (ms) | Pandas (ms) | Faster Engine |
|---|---:|---:|---|
| `medium-csv-stream-read` | 2.205166 ± 0.132996 | 12.433833 ± 0.437989 | SwiftSci 5.64× |
| `medium-csv-stream-filter` | 2.590458 ± 0.058889 | 14.074625 ± 0.311707 | SwiftSci 5.43× |
| `medium-csv-stream-group` | 2.911000 ± 0.340554 | 22.965709 ± 2.095836 | SwiftSci 7.89× |
| `medium-group-sum-mean` | 0.484791 ± 0.021971 | 1.544625 ± 0.029112 | SwiftSci 3.19× |
| `medium-row-sum` | 3.517458 ± 0.090340 | 19.108917 ± 38.793769 | SwiftSci 5.43× |
| `medium-inner-join` | 1.998042 ± 0.123445 | 0.673959 ± 0.213735 | Pandas 2.96× |
| `medium-pearson` | 0.162167 ± 0.011236 | 0.246000 ± 0.007099 | SwiftSci 1.52× |
| `medium-spearman` | 2.957583 ± 0.097745 | 3.465166 ± 0.085334 | SwiftSci 1.17× |
| `mixed-medium-csv-read` | 7.896333 ± 0.742763 | 13.809084 ± 0.196653 | SwiftSci 1.75× |
| `mixed-medium-csv-stream-read` | 8.015833 ± 0.261099 | 16.649875 ± 0.114735 | SwiftSci 2.08× |
| `mixed-medium-csv-stream-filter` | 8.920625 ± 0.553333 | 18.549333 ± 0.336056 | SwiftSci 2.08× |
| `mixed-medium-csv-stream-group` | 10.704833 ± 0.287987 | 29.750709 ± 1.218366 | SwiftSci 2.78× |
| `mixed-medium-filter` | 0.278041 ± 0.027278 | 0.832375 ± 0.027744 | SwiftSci 2.99× |
| `mixed-medium-group-sum-mean` | 1.892875 ± 0.058373 | 2.148875 ± 0.069925 | SwiftSci 1.14× |
| `mixed-medium-sort` | 2.271250 ± 0.059937 | 3.188625 ± 0.095844 | SwiftSci 1.40× |
| `mixed-medium-flat-matrix` | 0.104416 ± 0.004463 | 0.649042 ± 0.031636 | SwiftSci 6.22× |
| `mixed-medium-row-sum` | 3.881208 ± 0.102144 | 46.560750 ± 0.383673 | SwiftSci 12.00× |
| `mixed-medium-inner-join` | 2.026416 ± 0.039350 | 0.842417 ± 0.023901 | Pandas 2.41× |
| `mixed-medium-parquet-write` | 7.702792 ± 0.055825 | 7.766250 ± 0.133657 | SwiftSci 1.01× |
| `mixed-medium-standard-scale` | 1.896250 ± 0.045248 | 0.849500 ± 0.024767 | Pandas 2.23× |
| `mixed-medium-minmax-scale` | 0.725292 ± 0.014792 | 0.762208 ± 0.035972 | SwiftSci 1.05× |
| `parquet-medium-read` | 23.078375 ± 0.073550 | 1.836750 ± 0.088079 | Pandas 12.56× |
| `medium-welch` | 0.294291 ± 0.007630 | 0.672000 ± 0.041680 | SwiftSci 2.28× |
| `medium-regression-metrics` | 0.906708 ± 0.027395 | 0.313333 ± 0.008998 | Pandas 2.89× |
| `medium-roc-auc` | 2.095916 ± 0.033627 | 2.030375 ± 0.060288 | Pandas 1.03× |
| `medium-onehot` | 14.927875 ± 0.449188 | 3.583958 ± 0.107861 | Pandas 4.17× |
| `medium-tfidf` | 100.293583 ± 0.633031 | 114.137583 ± 2.450613 | SwiftSci 1.14× |
| `tiny-sqlite-ingest` | 0.058708 ± 0.026355 | 0.121208 ± 0.056249 | SwiftSci 2.06× |
| `tiny-rag-summary` | unresolved (0.000584) | 0.016291 ± 0.002703 | — |
| `tiny-pool-dice` | 0.002791 ± 0.000305 | 0.008208 ± 0.001572 | SwiftSci 2.94× |
| `large-mean` | 0.078625 ± 0.001518 | 0.118416 ± 0.001086 | SwiftSci 1.51× |
| `large-variance` | 0.523500 ± 0.008289 | 0.480250 ± 0.007191 | Pandas 1.09× |
| `large-stddev` | 0.523917 ± 0.006690 | 0.480542 ± 0.004401 | Pandas 1.09× |
| `medium-student` | 0.286292 ± 0.010500 | 0.671083 ± 0.013820 | SwiftSci 2.34× |
| `medium-paired` | 0.282292 ± 0.010962 | 0.545125 ± 0.008950 | SwiftSci 1.93× |
| `medium-anova` | 0.438625 ± 0.017600 | 0.845875 ± 0.022469 | SwiftSci 1.93× |

## NIST profile (`nist`, 27 cases)

| Case | SwiftSci (ms) | Pandas (ms) | Polars (ms) | DuckDB (ms) | SwiftSci vs Pandas | SwiftSci vs Polars | SwiftSci vs DuckDB |
|---|---:|---:|---:|---:|---|---|---|
| `nist-pidigits-mean` | unresolved (0.000416) | 0.004041 ± 0.000116 | 0.013417 ± 0.000748 | 0.141166 ± 0.019401 | — | — | — |
| `nist-pidigits-variance` | 0.003208 ± 0.000153 | 0.009791 ± 0.000211 | 0.013916 ± 0.000687 | 0.167000 ± 0.015709 | SwiftSci 3.05× | SwiftSci 4.34× | SwiftSci 52.06× |
| `nist-pidigits-stddev` | 0.002875 ± 0.000098 | 0.010334 ± 0.000706 | 0.015792 ± 0.000428 | 0.151625 ± 0.014776 | SwiftSci 3.59× | SwiftSci 5.49× | SwiftSci 52.74× |
| `nist-lottery-mean` | unresolved (0.000167) | 0.003208 ± 0.000090 | 0.012458 ± 0.002924 | 0.115333 ± 0.012712 | — | — | — |
| `nist-lottery-variance` | unresolved (0.000375) | 0.006375 ± 0.000419 | 0.012958 ± 0.000935 | 0.111417 ± 0.023659 | — | — | — |
| `nist-lottery-stddev` | unresolved (0.000334) | 0.006459 ± 0.000387 | 0.012000 ± 0.001143 | 0.134959 ± 0.024940 | — | — | — |
| `nist-lew-mean` | unresolved (0.000167) | 0.003167 ± 0.000194 | 0.011667 ± 0.000886 | 0.109583 ± 0.005185 | — | — | — |
| `nist-lew-variance` | unresolved (0.000333) | 0.006000 ± 0.000381 | 0.012750 ± 0.000960 | 0.117708 ± 0.015202 | — | — | — |
| `nist-lew-stddev` | unresolved (0.000375) | 0.007084 ± 0.000401 | 0.013084 ± 0.000996 | 0.106542 ± 0.008204 | — | — | — |
| `nist-mavro-mean` | unresolved (0.000125) | 0.003083 ± 0.000171 | 0.012834 ± 0.000640 | 0.120958 ± 0.007178 | — | — | — |
| `nist-mavro-variance` | unresolved (0.000291) | 0.006500 ± 0.000361 | 0.011750 ± 0.000828 | 0.109250 ± 0.016113 | — | — | — |
| `nist-mavro-stddev` | unresolved (0.000250) | 0.006792 ± 0.000425 | 0.012792 ± 0.000594 | 0.109584 ± 0.011999 | — | — | — |
| `nist-michelso-mean` | unresolved (0.000125) | 0.003250 ± 0.000192 | 0.013709 ± 0.001129 | 0.129667 ± 0.018120 | — | — | — |
| `nist-michelso-variance` | unresolved (0.000292) | 0.006750 ± 0.000256 | 0.013208 ± 0.000473 | 0.132209 ± 0.015308 | — | — | — |
| `nist-michelso-stddev` | unresolved (0.000292) | 0.007125 ± 0.000299 | 0.013125 ± 0.004181 | 0.124333 ± 0.015537 | — | — | — |
| `nist-numacc1-mean` | unresolved (0.000167) | 0.003125 ± 0.000148 | 0.012708 ± 0.000598 | 0.112458 ± 0.004216 | — | — | — |
| `nist-numacc1-variance` | unresolved (0.000208) | 0.006500 ± 0.000118 | 0.011917 ± 0.000704 | 0.111833 ± 0.014327 | — | — | — |
| `nist-numacc1-stddev` | unresolved (0.000209) | 0.006792 ± 0.000157 | 0.013125 ± 0.001206 | 0.125500 ± 0.020885 | — | — | — |
| `nist-numacc2-mean` | unresolved (0.000208) | 0.003125 ± 0.000171 | 0.012250 ± 0.000405 | 0.115375 ± 0.012183 | — | — | — |
| `nist-numacc2-variance` | unresolved (0.000791) | 0.007333 ± 0.000541 | 0.013458 ± 0.000697 | 0.132916 ± 0.011273 | — | — | — |
| `nist-numacc2-stddev` | unresolved (0.000792) | 0.007792 ± 0.000313 | 0.013375 ± 0.004231 | 0.123458 ± 0.020731 | — | — | — |
| `nist-numacc3-mean` | unresolved (0.000208) | 0.003375 ± 0.000082 | 0.012916 ± 0.000454 | 0.118916 ± 0.013445 | — | — | — |
| `nist-numacc3-variance` | unresolved (0.000750) | 0.007250 ± 0.000642 | 0.013625 ± 0.000752 | 0.123750 ± 0.004349 | — | — | — |
| `nist-numacc3-stddev` | unresolved (0.000834) | 0.007750 ± 0.000124 | 0.013167 ± 0.000907 | 0.125666 ± 0.007327 | — | — | — |
| `nist-numacc4-mean` | unresolved (0.000167) | 0.003083 ± 0.000217 | 0.012292 ± 0.000758 | 0.107500 ± 0.021870 | — | — | — |
| `nist-numacc4-variance` | unresolved (0.000792) | 0.007334 ± 0.000355 | 0.013458 ± 0.000127 | 0.144959 ± 0.019574 | — | — | — |
| `nist-numacc4-stddev` | unresolved (0.000792) | 0.007667 ± 0.000444 | 0.013125 ± 0.000727 | 0.115625 ± 0.003973 | — | — | — |

## 768-D `MLPClassifier` Mini-Batch BLAS Benchmark (`MLPBench`)

Standalone verified benchmark (`N = 512, D = 768, C = 5`, `hiddenLayerSizes = [64]`, `batchSize = 32`, `maxIter = 20`, 5 timed trials, `train_accuracy = 1.0000` on both engines):

| Engine | Stage | Median (ms) | Raw Samples (ms) | Faster Engine |
|---|---|---:|---|---|
| SwiftSci 3.13.3 | `fit` | `40.312` | `[40.957, 43.450, 39.742, 39.449, 40.312]` | **SwiftSci 2.57×** |
| Scikit-Learn 1.9.1 | `fit` | `103.639` | `[96.258, 103.639, 104.104, 107.793, 102.977]` | — |
| SwiftSci 3.13.3 | `predictProbability` | `0.504` | — | — |
| Scikit-Learn 1.9.1 | `predict_proba` | `0.444` | — | **Python 1.14×** |

## Scientific comparisons (`39/39` workloads, AC `legacy-capped`)

Five independent processes per workload with two warmups and five measured samples each (`Python 3.14.6`, `numpy 2.5.3`, `pandas 3.0.6`, `scipy 1.18.1`, `scikit-learn 1.9.1`, `statsmodels 0.15.0`, `shap 0.54.1`). Cross-engine ratios below are limited to stages cleared by scientific accuracy and comparability checks (`comparable_estimator: true`, excluding timing-resolution-limited `< 1 µs` stages such as `ts_hw_syn500_fixed` infer at `0.000750 ms`).

| Workload | Stage | SwiftSci median (IQR), ms | Python median (IQR), ms | Faster engine | Factor |
|---|---|---:|---:|---|---:|
| `ts_hw_syn500_fixed` | fit | 0.010333 (0.000374) | 1.704750 (0.013749) | SwiftSci | 164.98× |
| `ts_hw_syn50k_fixed` | fit | 1.177500 (0.010208) | 135.612500 (1.890625) | SwiftSci | 115.17× |
| `ts_hw_syn50k_fixed` | infer | 0.052000 (0.002583) | 133.891083 (3.380500) | SwiftSci | 2574.83× |
| `ts_arima_syn500_matched_hr` | fit | 0.038625 (0.000583) | 3.142125 (0.018959) | SwiftSci | 81.35× |
| `ts_arima_syn500_matched_hr` | infer | 0.001292 (0.000001) | 0.302334 (0.002875) | SwiftSci | 234.00× |
| `ts_arima_dgp_syn50k_matched_hr` | fit | 2.522250 (0.128334) | 202.780000 (2.464375) | SwiftSci | 80.40× |
| `ts_arima_dgp_syn50k_matched_hr` | infer | 0.079541 (0.000875) | 0.343834 (0.009041) | SwiftSci | 4.32× |
| `ts_arima_dgp_syn500_matched_hr` | fit | 0.038667 (0.000750) | 3.129542 (0.027708) | SwiftSci | 80.94× |
| `ts_arima_dgp_syn500_matched_hr` | infer | 0.001291 (0.000001) | 0.305833 (0.003250) | SwiftSci | 236.90× |
| `ml_ols_regression` | fit | 0.440959 (0.000917) | 0.297167 (0.021291) | Python | 1.48× |
| `ml_ols_regression` | infer | 0.001458 (0.000541) | 0.033292 (0.000958) | SwiftSci | 22.83× |
| `ml_linear_svc` | fit | 12.009250 (0.368333) | 0.637917 (0.004417) | Python | 18.83× |
| `ml_linear_svc` | infer | 0.004375 (0.000251) | 0.093916 (0.002709) | SwiftSci | 21.47× |
| `ml_decision_tree_regressor` | fit | 1.754375 (0.019125) | 2.885042 (0.050083) | SwiftSci | 1.64× |
| `ml_decision_tree_regressor` | infer | 0.003833 (0.000208) | 0.038791 (0.000209) | SwiftSci | 10.12× |
| `ml_decision_tree_classifier` | fit | 3.721833 (0.176791) | 3.306583 (0.036667) | Python | 1.13× |
| `ml_decision_tree_classifier` | infer | 0.011584 (0.000916) | 0.093833 (0.000500) | SwiftSci | 8.10× |
| `ml_random_forest_regressor` | fit | 9.981792 (0.131250) | 108.545125 (0.173917) | SwiftSci | 10.87× |
| `ml_random_forest_regressor` | infer | 0.216292 (0.008750) | 1.647458 (0.002584) | SwiftSci | 7.62× |
| `ml_random_forest_classifier` | fit | 15.085750 (0.156791) | 51.680458 (0.699126) | SwiftSci | 3.43× |
| `ml_random_forest_classifier` | infer | 1.058458 (0.154458) | 3.453459 (0.036876) | SwiftSci | 3.26× |
| `ml_gbdt_regressor` | fit | 20.483125 (0.090959) | 99.483625 (0.358542) | SwiftSci | 4.86× |
| `ml_gbdt_regressor` | infer | 0.054791 (0.000541) | 0.201792 (0.004583) | SwiftSci | 3.68× |
| `ml_hist_gbdt_regressor` | fit | 12.565958 (0.071583) | 17.880166 (0.108208) | SwiftSci | 1.42× |
| `ml_hist_gbdt_regressor` | infer | 0.196959 (0.002126) | 0.414958 (0.004792) | SwiftSci | 2.11× |
| `ml_hist_gbdt_classifier` | fit | 14.608041 (0.030001) | 23.382708 (0.084416) | SwiftSci | 1.60× |
| `ml_hist_gbdt_classifier` | infer | 0.476250 (0.009708) | 1.250875 (0.052500) | SwiftSci | 2.63× |
| `cluster_pca` | fit | 0.146416 (0.000084) | 0.256417 (0.013542) | SwiftSci | 1.75× |
| `cluster_pca` | infer | 0.069791 (0.004209) | 0.058875 (0.001499) | Python | 1.19× |
| `cluster_kmeans` | fit | 0.205459 (0.001916) | 0.491958 (0.005250) | SwiftSci | 2.39× |
| `cluster_kmeans` | infer | 0.038000 (0.002250) | 0.073750 (0.002041) | SwiftSci | 1.94× |
| `stats_ddof1_100k` | fit | 1.129208 (0.003667) | 4.113584 (0.046584) | SwiftSci | 3.64× |
| `stats_ddof1_100k` | infer | 1.129208 (0.003667) | 4.113584 (0.046584) | SwiftSci | 3.64× |

Unmatched diagnostic disclosure: `ts_hw_syn500_opt` fit measured `0.271333 ms` (IQR `0.001417 ms`) for SwiftSci vs `48.074250 ms` (IQR `0.337000 ms`) for statsmodels; `ts_hw_syn500_fixed` infer measured `0.000750 ms` (IQR `0.000042 ms`, `< 1 µs` resolution-limited) vs `1.484708 ms` (IQR `0.011792 ms`).

### SwiftLLM KV-cache ablation

| Workload | Full forward (ms) | KV cache (ms) | Full forward / KV cache |
|---|---:|---:|---:|
| `llm_kvcache_ablation_p24_d16` | 6.448291 | 5.770625 | 1.12× |
| `llm_kvcache_ablation_p64_d24` | 9.937875 | 8.655833 | 1.15× |

## Interpretation and reproduction

Compare engines using the same input, output contract, estimator, and execution policy. `v3.13.3-comparison` ran under `production-default` (AC power, native thread pools); `scientific_validation.json` ran under `legacy-capped` (AC power). Peak RSS in raw files includes runtime/library startup allocations.

Start at the [runner guide](../../README.md), review the [evidence index](README.md), and see the [deep accuracy audit](accuracy.deep.md) for acceptance and comparator scope.
