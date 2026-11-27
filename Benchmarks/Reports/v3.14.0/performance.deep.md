# SwiftSci 3.14.0 Deep Performance Audit

> **Navigation**: [← v3.14.0 Evidence Index](README.md) · [Performance Overview](../../../PERFORMANCE/3.14.0.md) · [Deep Accuracy Audit](accuracy.deep.md) · [Accuracy Overview](../../../ACCURACY/3.14.0.md)

Branch: `release/3.14.0` · Base commit: `10c11b0c586663484e0f982f7afd52834d3924d8` (`dirty: true` with `3.14.0` changes)

- Source tree SHA-256 (`summary.json` / `suite.json`): `c6cd7882049ccb65c485c5d113880a44615a5fbb553137ef13abe7bfb79fa2b7`
- Benchmark worker SHA-256 (`SwiftSciBenchmarkWorker`, on disk & `scientific_validation.json`): `0f75343624dbaf306c452262adb5e675b2436cf60aa538e505b336edd74e9968`
- Scientific validation JSON SHA-256: `e0d38b0d9aa527d36ca6b84bde511278014e0f7ca8bdd8b5c7a8f9e8cb22dbdc`
- Release-gate `acceptance.json` SHA-256: `d09e640e37638601761cf36829e36d193232011bb28551c19f2828f5dc2df15b`
- Release-gate `summary.json` SHA-256: `7e771fa8d48b96a3b942bf7ac195f5c2ac94d3beafbd0ed5913e759d8273aefa`
- Comparison `suite.json` SHA-256: `ab111e4068d067e21931894e0c7465b8d0575f21a1b3d7cbfe67a543af215222`
- Comparison `measurements.json` SHA-256: `751a7d3c63905827db358c560ddcbaade8462a80a076cd78177cf75e6e5a0eea`
- Comparison `cpu-acceptance/acceptance.json` SHA-256: `f28fb2282ed026b6804a4011e907d2fb42852cf6f88467f89330858ebb21d5da`
- `v3140_bench` Release binary SHA-256: `5c8248e16f8de95e6273d494038d22b3d043194b9c564e86b8447eba12ce5806`
- `v3140_bench/bench_data.json` SHA-256: `76fc5dab4f6456e739c6e2e74e503f072d8630bcfb49701ca3ab33c6179eb349`
- [Release evidence index](README.md)

Apple M4 Pro (`arm64`), 24 GiB RAM (`25,769,803,776` bytes), `macOS 27.2.1` build `26B131`, AC power (`power_start = AC`, `power_end = AC`, `power_unchanged = true` across all 6 `v3.14.0-release-gate/comparison` profiles and `scientific_validation.json`). Xcode `27.2 (27B52)`, Swift `6.4.2`. The Release worker was verified free of coverage instrumentation and built with `MTL_FAST_MATH=NO`, `MTL_MATH_MODE=SAFE`.

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
| `medium-csv-read` | 2.404750 ± 0.064495 | 10.353459 ± 0.082633 | 1.052875 ± 0.125630 | 25.674000 ± 0.199070 | SwiftSci 4.31× | Polars 2.28× | SwiftSci 10.68× |
| `medium-filter` | 0.230375 ± 0.017286 | 0.731417 ± 0.071896 | 0.667042 ± 0.015053 | 0.989833 ± 0.026679 | SwiftSci 3.17× | SwiftSci 2.90× | SwiftSci 4.30× |
| `medium-sort` | 2.219375 ± 0.061888 | 2.737292 ± 0.068441 | 1.558666 ± 0.193887 | 6.051042 ± 0.088881 | SwiftSci 1.23× | Polars 1.42× | SwiftSci 2.73× |
| `medium-group-sum` | 0.373250 ± 0.015429 | 0.515250 ± 0.045879 | 0.577458 ± 0.058085 | 0.299500 ± 0.027003 | SwiftSci 1.38× | SwiftSci 1.55× | DuckDB 1.25× |
| `medium-flat-matrix` | 0.106083 ± 0.008827 | 0.566750 ± 0.040877 | 1.448000 ± 0.044735 | 1.454500 ± 0.072713 | SwiftSci 5.34× | SwiftSci 13.65× | SwiftSci 13.71× |
| `medium-target` | 0.055542 ± 0.004903 | 0.052958 ± 0.036548 | 0.048000 ± 0.005569 | 1.035000 ± 0.091061 | Pandas 1.05× | Polars 1.16× | SwiftSci 18.63× |
| `medium-standard-scale` | 2.032792 ± 0.052908 | 0.841500 ± 0.098093 | 1.503667 ± 0.044112 | 2.408500 ± 0.070547 | Pandas 2.42× | Polars 1.35× | SwiftSci 1.18× |
| `medium-minmax-scale` | 0.775333 ± 0.021853 | 0.756917 ± 0.057033 | 1.748500 ± 0.067055 | 2.564292 ± 0.137844 | Pandas 1.02× | SwiftSci 2.26× | SwiftSci 3.31× |
| `medium-mean` | 0.008000 ± 0.000458 | 0.014917 ± 0.000717 | 0.027208 ± 0.002524 | 0.178042 ± 0.020094 | SwiftSci 1.86× | SwiftSci 3.40× | SwiftSci 22.26× |
| `medium-variance` | 0.054917 ± 0.003931 | 0.055500 ± 0.000253 | 0.059000 ± 0.003010 | 0.487125 ± 0.027020 | SwiftSci 1.01× | SwiftSci 1.07× | SwiftSci 8.87× |
| `medium-stddev` | 0.052709 ± 0.002887 | 0.055708 ± 0.000755 | 0.061000 ± 0.002943 | 0.492333 ± 0.025533 | SwiftSci 1.06× | SwiftSci 1.16× | SwiftSci 9.34× |
| `nist-mean` | unresolved (0.000167) | 0.003417 ± 0.000413 | 0.012000 ± 0.000706 | 0.113750 ± 0.011364 | — | — | — |
| `nist-variance` | unresolved (0.000833) | 0.007208 ± 0.000533 | 0.012084 ± 0.000449 | 0.118792 ± 0.015728 | — | — | — |
| `nist-stddev` | unresolved (0.000750) | 0.007500 ± 0.000418 | 0.013834 ± 0.000903 | 0.133458 ± 0.009990 | — | — | — |

## Extended profile (`extended`, 1M rows)

| Case | SwiftSci (ms) | Pandas (ms) | Polars (ms) | DuckDB (ms) | SwiftSci vs Pandas | SwiftSci vs Polars | SwiftSci vs DuckDB |
|---|---:|---:|---:|---:|---|---|---|
| `large-csv-read` | 22.835958 ± 0.629635 | 92.501833 ± 2.108081 | 6.226041 ± 0.318740 | 53.920375 ± 0.603796 | SwiftSci 4.05× | Polars 3.67× | SwiftSci 2.36× |
| `large-filter` | 2.088583 ± 0.022440 | 4.582084 ± 0.143186 | 1.004291 ± 0.089490 | 5.981667 ± 0.083985 | SwiftSci 2.19× | Polars 2.08× | SwiftSci 2.86× |
| `large-sort` | 23.059875 ± 0.649656 | 31.492292 ± 0.340884 | 10.878583 ± 1.083948 | 20.614000 ± 0.984115 | SwiftSci 1.37× | Polars 2.12× | DuckDB 1.12× |
| `large-group-sum` | 4.115917 ± 0.047988 | 3.971792 ± 0.063570 | 2.834209 ± 0.096997 | 0.701125 ± 0.051698 | Pandas 1.04× | Polars 1.45× | DuckDB 5.87× |
| `large-flat-matrix` | 1.068750 ± 0.092967 | 2.837833 ± 0.083918 | 2.128334 ± 0.211340 | 6.005000 ± 0.188654 | SwiftSci 2.66× | SwiftSci 1.99× | SwiftSci 5.62× |
| `large-target` | 0.518375 ± 0.054581 | 0.239792 ± 0.012123 | 0.613709 ± 0.049220 | 4.057625 ± 0.150359 | Pandas 2.16× | SwiftSci 1.18× | SwiftSci 7.83× |
| `large-standard-scale` | 18.470167 ± 2.315613 | 4.419041 ± 0.147446 | 4.348541 ± 0.402323 | 10.244208 ± 0.501509 | Pandas 4.18× | Polars 4.25× | DuckDB 1.80× |
| `large-minmax-scale` | 9.347375 ± 0.945422 | 3.625542 ± 0.034404 | 4.286166 ± 0.459045 | 10.280917 ± 1.019525 | Pandas 2.58× | Polars 2.18× | SwiftSci 1.10× |
| `large-mean` | 0.083791 ± 0.005631 | 0.120291 ± 0.008527 | 0.120625 ± 0.014460 | 0.313000 ± 0.023151 | SwiftSci 1.44× | SwiftSci 1.44× | SwiftSci 3.74× |
| `large-variance` | 0.611958 ± 0.035790 | 0.529875 ± 0.055883 | 0.522333 ± 0.045222 | 0.810458 ± 0.140009 | Pandas 1.15× | Polars 1.17× | SwiftSci 1.32× |
| `large-stddev` | 0.505083 ± 0.021747 | 0.436792 ± 0.010395 | 0.410541 ± 0.004452 | 0.793416 ± 0.125015 | Pandas 1.16× | Polars 1.23× | SwiftSci 1.57× |
| `nist-mean` | unresolved (0.000167) | 0.003125 ± 0.000059 | 0.012000 ± 0.001068 | 0.119041 ± 0.027460 | — | — | — |
| `nist-variance` | unresolved (0.000792) | 0.006750 ± 0.001675 | 0.013625 ± 0.000766 | 0.135959 ± 0.017659 | — | — | — |
| `nist-stddev` | unresolved (0.000791) | 0.006917 ± 0.000589 | 0.012542 ± 0.000373 | 0.151750 ± 0.020428 | — | — | — |

## Public-data profile (`public-data`)

| Case | SwiftSci (ms) | Pandas (ms) | Polars (ms) | DuckDB (ms) | SwiftSci vs Pandas | SwiftSci vs Polars | SwiftSci vs DuckDB |
|---|---:|---:|---:|---:|---|---|---|
| `medium-uniform-h2o-q1` | 1.777083 ± 0.066420 | 1.095000 ± 0.020400 | 1.146125 ± 0.093618 | 1.402917 ± 0.059722 | Pandas 1.62× | Polars 1.55× | DuckDB 1.27× |
| `medium-uniform-h2o-q2` | 6.612917 ± 0.176538 | 3.038625 ± 0.178711 | 1.869875 ± 0.048894 | 2.604375 ± 0.165030 | Pandas 2.18× | Polars 3.54× | DuckDB 2.54× |
| `medium-uniform-h2o-q3` | 2.336875 ± 0.124209 | 1.693916 ± 0.184655 | 1.169208 ± 0.095207 | 1.452209 ± 0.088371 | Pandas 1.38× | Polars 2.00× | DuckDB 1.61× |
| `medium-uniform-h2o-q4` | 0.568709 ± 0.035374 | 0.981167 ± 0.014179 | 0.939917 ± 0.063493 | 0.494208 ± 0.043616 | SwiftSci 1.73× | SwiftSci 1.65× | DuckDB 1.15× |
| `medium-uniform-h2o-q5` | 0.790916 ± 0.112515 | 1.059166 ± 0.276130 | 1.001250 ± 0.306452 | 0.465667 ± 0.036470 | SwiftSci 1.34× | SwiftSci 1.27× | DuckDB 1.70× |
| `medium-hot-key-h2o-q1` | 1.671708 ± 0.041915 | 1.068625 ± 0.015998 | 1.180875 ± 0.096297 | 1.385709 ± 0.027180 | Pandas 1.56× | Polars 1.42× | DuckDB 1.21× |
| `medium-hot-key-h2o-q2` | 5.249458 ± 0.090108 | 2.444625 ± 0.263759 | 2.019458 ± 0.158248 | 2.356000 ± 0.190995 | Pandas 2.15× | Polars 2.60× | DuckDB 2.23× |
| `medium-hot-key-h2o-q3` | 2.192792 ± 0.144344 | 1.705041 ± 0.091138 | 1.717792 ± 0.036609 | 1.786250 ± 0.110391 | Pandas 1.29× | Polars 1.28× | DuckDB 1.23× |
| `medium-hot-key-h2o-q4` | 1.106167 ± 0.039038 | 2.011500 ± 0.025447 | 1.200084 ± 0.022973 | 0.954667 ± 0.027701 | SwiftSci 1.82× | SwiftSci 1.08× | DuckDB 1.16× |
| `medium-hot-key-h2o-q5` | 1.270833 ± 0.045260 | 1.597917 ± 0.054450 | 1.323875 ± 0.026481 | 0.960750 ± 0.016478 | SwiftSci 1.26× | SwiftSci 1.04× | DuckDB 1.32× |
| `wine-red-pipeline` | 0.378500 ± 0.015000 | 1.407458 ± 0.051814 | 1.212500 ± 0.039559 | 18.037708 ± 0.635660 | SwiftSci 3.72× | SwiftSci 3.20× | SwiftSci 47.66× |

## Tabular-migration profile (`tabular-migration`)

| Case | SwiftSci (ms) | Pandas (ms) | Polars (ms) | DuckDB (ms) | SwiftSci vs Pandas | SwiftSci vs Polars | SwiftSci vs DuckDB |
|---|---:|---:|---:|---:|---|---|---|
| `medium-group-sum-mean` | 0.462125 ± 0.014901 | 1.469709 ± 0.025539 | 0.660833 ± 0.027493 | 0.368541 ± 0.033957 | SwiftSci 3.18× | SwiftSci 1.43× | DuckDB 1.25× |
| `medium-row-sum` | 3.480458 ± 0.173576 | 18.242292 ± 0.162169 | 22.273375 ± 0.367530 | 15.656417 ± 0.265158 | SwiftSci 5.24× | SwiftSci 6.40× | SwiftSci 4.50× |
| `medium-inner-join` | 1.996958 ± 0.125371 | 0.777041 ± 0.166185 | 1.796667 ± 0.096193 | 1.975208 ± 0.085521 | Pandas 2.57× | Polars 1.11× | DuckDB 1.01× |
| `medium-pearson` | 0.147625 ± 0.005630 | 0.240791 ± 0.013123 | 0.147958 ± 0.017816 | 0.503750 ± 0.021950 | SwiftSci 1.63× | SwiftSci 1.00× | SwiftSci 3.41× |
| `medium-spearman` | 2.984666 ± 0.166298 | 3.159541 ± 0.090305 | 2.736541 ± 0.191984 | 9.999750 ± 0.079316 | SwiftSci 1.06× | Polars 1.09× | SwiftSci 3.35× |
| `mixed-medium-csv-read` | 7.745792 ± 0.316340 | 13.183041 ± 0.062890 | 1.560458 ± 0.035860 | 34.248625 ± 0.495302 | SwiftSci 1.70× | Polars 4.96× | SwiftSci 4.42× |
| `mixed-medium-filter` | 0.275125 ± 0.017711 | 0.798458 ± 0.146776 | 0.711167 ± 0.056145 | 1.267458 ± 0.134154 | SwiftSci 2.90× | SwiftSci 2.58× | SwiftSci 4.61× |
| `mixed-medium-group-sum-mean` | 2.075167 ± 0.120415 | 1.890209 ± 0.021066 | 0.815042 ± 0.060955 | 1.165042 ± 0.064847 | Pandas 1.10× | Polars 2.55× | DuckDB 1.78× |
| `mixed-medium-sort` | 2.157833 ± 0.059301 | 2.840959 ± 0.266593 | 1.567541 ± 0.181854 | 6.528833 ± 0.278202 | SwiftSci 1.32× | Polars 1.38× | SwiftSci 3.03× |
| `mixed-medium-flat-matrix` | 0.099750 ± 0.004273 | 0.657417 ± 0.156635 | 1.407208 ± 0.113665 | 1.386083 ± 0.184471 | SwiftSci 6.59× | SwiftSci 14.11× | SwiftSci 13.90× |
| `mixed-medium-row-sum` | 3.495666 ± 0.091219 | 46.808916 ± 2.354408 | 26.189000 ± 0.238133 | 22.534208 ± 0.230307 | SwiftSci 13.39× | SwiftSci 7.49× | SwiftSci 6.45× |
| `mixed-medium-inner-join` | 1.860208 ± 0.026485 | 0.719125 ± 0.074905 | 1.984541 ± 0.053876 | 2.715250 ± 0.082920 | Pandas 2.59× | SwiftSci 1.07× | SwiftSci 1.46× |
| `mixed-medium-parquet-write` | 7.337833 ± 0.511518 | 7.270833 ± 0.221384 | 1.531292 ± 0.102648 | 3.614500 ± 0.244194 | Pandas 1.01× | Polars 4.79× | DuckDB 2.03× |
| `mixed-medium-standard-scale` | 1.974500 ± 0.090852 | 0.936750 ± 0.209924 | 1.436000 ± 0.083481 | 2.525250 ± 0.184552 | Pandas 2.11× | Polars 1.38× | SwiftSci 1.28× |
| `mixed-medium-minmax-scale` | 0.750500 ± 0.036544 | 0.779167 ± 0.039286 | 1.756541 ± 0.122240 | 2.804917 ± 0.121878 | SwiftSci 1.04× | SwiftSci 2.34× | SwiftSci 3.74× |
| `parquet-medium-read` | 23.084584 ± 0.525352 | 1.978875 ± 0.060148 | 1.997584 ± 0.048073 | 2.947208 ± 0.084053 | Pandas 11.67× | Polars 11.56× | DuckDB 7.83× |
| `large-mean` | 0.079417 ± 0.003317 | 0.118042 ± 0.001915 | 0.103750 ± 0.034876 | 0.311375 ± 0.022520 | SwiftSci 1.49× | SwiftSci 1.31× | SwiftSci 3.92× |
| `large-variance` | 0.532334 ± 0.005657 | 0.494041 ± 0.006042 | 0.478792 ± 0.025186 | 0.825875 ± 0.112632 | Pandas 1.08× | Polars 1.11× | SwiftSci 1.55× |
| `large-stddev` | 0.536750 ± 0.009398 | 0.492167 ± 0.012630 | 0.456667 ± 0.010360 | 0.836375 ± 0.076102 | Pandas 1.09× | Polars 1.18× | SwiftSci 1.56× |

## Migration profile (`migration`, 36 cases)

| Case | SwiftSci (ms) | Pandas (ms) | Faster Engine |
|---|---:|---:|---|
| `medium-csv-stream-read` | 2.282917 ± 0.065928 | 12.862625 ± 0.135061 | SwiftSci 5.63× |
| `medium-csv-stream-filter` | 2.666417 ± 0.141339 | 14.955291 ± 0.767290 | SwiftSci 5.61× |
| `medium-csv-stream-group` | 3.143833 ± 0.233145 | 26.655375 ± 2.147482 | SwiftSci 8.48× |
| `medium-group-sum-mean` | 0.533625 ± 0.025016 | 1.662792 ± 0.104900 | SwiftSci 3.12× |
| `medium-row-sum` | 3.980958 ± 0.126929 | 19.189709 ± 0.067232 | SwiftSci 4.82× |
| `medium-inner-join` | 2.207041 ± 0.183087 | 0.902750 ± 0.073179 | Pandas 2.44× |
| `medium-pearson` | 0.163417 ± 0.013763 | 0.261709 ± 0.026585 | SwiftSci 1.60× |
| `medium-spearman` | 3.404750 ± 0.124573 | 3.945334 ± 0.343444 | SwiftSci 1.16× |
| `mixed-medium-csv-read` | 9.935708 ± 1.132061 | 16.937833 ± 1.339952 | SwiftSci 1.70× |
| `mixed-medium-csv-stream-read` | 8.396625 ± 0.734963 | 17.165709 ± 0.324170 | SwiftSci 2.04× |
| `mixed-medium-csv-stream-filter` | 8.673000 ± 0.398058 | 16.404458 ± 3.101471 | SwiftSci 1.89× |
| `mixed-medium-csv-stream-group` | 10.238708 ± 0.196031 | 25.989292 ± 0.355470 | SwiftSci 2.54× |
| `mixed-medium-filter` | 0.281042 ± 0.101840 | 0.628083 ± 0.023639 | SwiftSci 2.23× |
| `mixed-medium-group-sum-mean` | 2.127291 ± 0.380388 | 2.135667 ± 0.258420 | SwiftSci 1.00× |
| `mixed-medium-sort` | 5.377459 ± 1.201373 | 7.445917 ± 0.877376 | SwiftSci 1.38× |
| `mixed-medium-flat-matrix` | 0.355958 ± 0.074803 | 1.611000 ± 0.041755 | SwiftSci 4.53× |
| `mixed-medium-row-sum` | 8.296541 ± 1.740466 | 87.663667 ± 17.864746 | SwiftSci 10.57× |
| `mixed-medium-inner-join` | 3.152334 ± 0.860093 | 1.451916 ± 0.343504 | Pandas 2.17× |
| `mixed-medium-parquet-write` | 7.572584 ± 0.483694 | 9.302666 ± 3.110539 | SwiftSci 1.23× |
| `mixed-medium-standard-scale` | 1.828625 ± 0.613072 | 0.891542 ± 0.225051 | Pandas 2.05× |
| `mixed-medium-minmax-scale` | 0.801583 ± 0.094671 | 0.798375 ± 0.131302 | Pandas 1.00× |
| `parquet-medium-read` | 33.922250 ± 10.476647 | 4.174875 ± 1.727561 | Pandas 8.13× |
| `medium-welch` | 0.365541 ± 0.098520 | 0.820500 ± 0.119573 | SwiftSci 2.24× |
| `medium-regression-metrics` | 1.271667 ± 0.282208 | 0.453875 ± 0.135767 | Pandas 2.80× |
| `medium-roc-auc` | 2.218208 ± 0.656600 | 2.164083 ± 0.198685 | Pandas 1.03× |
| `medium-onehot` | 13.691125 ± 0.970218 | 3.318041 ± 0.139495 | Pandas 4.13× |
| `medium-tfidf` | 92.581959 ± 1.814130 | 101.836042 ± 4.147750 | SwiftSci 1.10× |
| `tiny-sqlite-ingest` | 0.053542 ± 0.003968 | 0.106708 ± 0.001326 | SwiftSci 1.99× |
| `tiny-rag-summary` | unresolved (0.000416) | 0.015834 ± 0.000664 | — |
| `tiny-pool-dice` | 0.003125 ± 0.000163 | 0.008375 ± 0.000537 | SwiftSci 2.68× |
| `large-mean` | 0.072708 ± 0.005016 | 0.109750 ± 0.005747 | SwiftSci 1.51× |
| `large-variance` | 0.485333 ± 0.031374 | 0.438750 ± 0.012307 | Pandas 1.11× |
| `large-stddev` | 0.480625 ± 0.035334 | 0.440042 ± 0.037619 | Pandas 1.09× |
| `medium-student` | 0.290750 ± 0.022395 | 0.622458 ± 0.092075 | SwiftSci 2.14× |
| `medium-paired` | 0.256542 ± 0.003473 | 0.487375 ± 0.035433 | SwiftSci 1.90× |
| `medium-anova` | 0.431875 ± 0.022567 | 0.757208 ± 0.055957 | SwiftSci 1.75× |

## NIST profile (`nist`, 27 cases)

| Case | SwiftSci (ms) | Pandas (ms) | Polars (ms) | DuckDB (ms) | SwiftSci vs Pandas | SwiftSci vs Polars | SwiftSci vs DuckDB |
|---|---:|---:|---:|---:|---|---|---|
| `nist-pidigits-mean` | unresolved (0.000375) | 0.003625 ± 0.000249 | 0.012834 ± 0.000860 | 0.125708 ± 0.024244 | — | — | — |
| `nist-pidigits-variance` | 0.002958 ± 0.000169 | 0.008917 ± 0.000760 | 0.014083 ± 0.000425 | 0.140000 ± 0.010297 | SwiftSci 3.01× | SwiftSci 4.76× | SwiftSci 47.33× |
| `nist-pidigits-stddev` | 0.002917 ± 0.000176 | 0.009958 ± 0.000485 | 0.014458 ± 0.000374 | 0.148083 ± 0.020980 | SwiftSci 3.41× | SwiftSci 4.96× | SwiftSci 50.77× |
| `nist-lottery-mean` | unresolved (0.000125) | 0.002916 ± 0.000197 | 0.012125 ± 0.000849 | 0.112875 ± 0.016525 | — | — | — |
| `nist-lottery-variance` | unresolved (0.000292) | 0.006167 ± 0.000178 | 0.012417 ± 0.000511 | 0.113875 ± 0.017044 | — | — | — |
| `nist-lottery-stddev` | unresolved (0.000333) | 0.006541 ± 0.000419 | 0.013416 ± 0.002036 | 0.127000 ± 0.042744 | — | — | — |
| `nist-lew-mean` | unresolved (0.000167) | 0.003250 ± 0.000231 | 0.013375 ± 0.000946 | 0.119375 ± 0.020264 | — | — | — |
| `nist-lew-variance` | unresolved (0.000291) | 0.005959 ± 0.000108 | 0.012042 ± 0.000979 | 0.107500 ± 0.005158 | — | — | — |
| `nist-lew-stddev` | unresolved (0.000333) | 0.006291 ± 0.000085 | 0.011666 ± 0.000219 | 0.123250 ± 0.011518 | — | — | — |
| `nist-mavro-mean` | unresolved (0.000125) | 0.002875 ± 0.000062 | 0.011667 ± 0.000733 | 0.124209 ± 0.014092 | — | — | — |
| `nist-mavro-variance` | unresolved (0.000250) | 0.005959 ± 0.000054 | 0.011833 ± 0.000213 | 0.113250 ± 0.013769 | — | — | — |
| `nist-mavro-stddev` | unresolved (0.000250) | 0.006334 ± 0.000130 | 0.012125 ± 0.001829 | 0.116708 ± 0.011230 | — | — | — |
| `nist-michelso-mean` | unresolved (0.000125) | 0.002959 ± 0.000199 | 0.011959 ± 0.000721 | 0.117333 ± 0.015982 | — | — | — |
| `nist-michelso-variance` | unresolved (0.000250) | 0.006875 ± 0.000567 | 0.013709 ± 0.000878 | 0.114708 ± 0.027408 | — | — | — |
| `nist-michelso-stddev` | unresolved (0.000250) | 0.006500 ± 0.002598 | 0.013125 ± 0.000616 | 0.110542 ± 0.007733 | — | — | — |
| `nist-numacc1-mean` | unresolved (0.000125) | 0.002917 ± 0.000225 | 0.012375 ± 0.000230 | 0.118792 ± 0.005933 | — | — | — |
| `nist-numacc1-variance` | unresolved (0.000167) | 0.005958 ± 0.000463 | 0.012042 ± 0.000271 | 0.115792 ± 0.008964 | — | — | — |
| `nist-numacc1-stddev` | unresolved (0.000208) | 0.006292 ± 0.000100 | 0.012000 ± 0.000476 | 0.113209 ± 0.016338 | — | — | — |
| `nist-numacc2-mean` | unresolved (0.000167) | 0.003125 ± 0.000076 | 0.011792 ± 0.000164 | 0.121792 ± 0.012314 | — | — | — |
| `nist-numacc2-variance` | unresolved (0.000750) | 0.007459 ± 0.000599 | 0.013917 ± 0.000713 | 0.125042 ± 0.017092 | — | — | — |
| `nist-numacc2-stddev` | unresolved (0.000791) | 0.009333 ± 0.001602 | 0.013875 ± 0.003030 | 0.153125 ± 0.029018 | — | — | — |
| `nist-numacc3-mean` | unresolved (0.000166) | 0.003583 ± 0.000093 | 0.013500 ± 0.002667 | 0.120125 ± 0.020060 | — | — | — |
| `nist-numacc3-variance` | unresolved (0.000791) | 0.008250 ± 0.000358 | 0.014916 ± 0.000868 | 0.137958 ± 0.014038 | — | — | — |
| `nist-numacc3-stddev` | unresolved (0.000833) | 0.008500 ± 0.000315 | 0.015333 ± 0.000272 | 0.124875 ± 0.022127 | — | — | — |
| `nist-numacc4-mean` | unresolved (0.000167) | 0.003458 ± 0.000261 | 0.013667 ± 0.001136 | 0.110209 ± 0.026056 | — | — | — |
| `nist-numacc4-variance` | unresolved (0.000833) | 0.008125 ± 0.000582 | 0.014917 ± 0.000831 | 0.153459 ± 0.026427 | — | — | — |
| `nist-numacc4-stddev` | unresolved (0.000833) | 0.008583 ± 0.000460 | 0.015042 ± 0.001330 | 0.147917 ± 0.022115 | — | — | — |

## 3.14.0 Feature Benchmarks (`v3140_bench`, Release)

Verified Wave 3 Release benchmark (`V3140Bench` vs `bench_python.py` using `scikit-learn 1.9.1` and `shap 0.54.1` on identical `bench_data.json` fixtures; `1` warmup / `5` rounds for TreeSHAP & `OneVsRestClassifier` fit; `2` warmups / `10` rounds for `predictProbability` / `predict_proba`):

| Workload ID | Description | SwiftSci 3.14.0 Median (ms) | Python Median (ms) | Faster Engine |
|---|---|---:|---:|---|
| `gbdt_treeshap_50x50_d4` | `GradientBoostedTreesRegressor` TreeSHAP (`50` trees, `depth=4`, `50` explain instances, `.empiricalCover`) | `3.709` | `1.045` (`shap`) | **Python 3.55×** |
| `rf_multiclass_treeshap_30x50_d5_c3` | `RandomForestClassifier` multiclass per-class TreeSHAP (`30` trees, `depth=5`, `3` classes, `50` explain instances, `.softLeafDistribution`) | `21.727` | `1.316` (`shap`) | **Python 16.51×** |
| `ovr_fit_1000x16_c4` | `OneVsRestClassifier` `fit` (`1000 × 16`, `4` classes, SwiftSci `1000` epochs vs `LogisticRegression(max_iter=1000)`) | `192.265` | `3.820` (`scikit-learn`) | **Python 50.33×** |
| `ovr_predict_proba_500x16_c4` | `OneVsRestClassifier` `predictProbability` / `predict_proba` (`500 × 16`, `4` classes) | `0.069` | `0.203` (`scikit-learn`) | **SwiftSci 2.94×** |

## Scientific comparisons (`39/39` workloads, AC `legacy-capped`)

Five independent processes per workload with two warmups and five measured samples each (`Python 3.14.6`, `numpy 2.5.3`, `pandas 3.0.6`, `scipy 1.18.1`, `scikit-learn 1.9.1`, `statsmodels 0.15.0`, `shap 0.54.1`). Cross-engine ratios below are limited to stages cleared by scientific accuracy and comparability checks (`comparable_estimator: true`, excluding timing-resolution-limited `< 1 µs` stages such as `ts_hw_syn500_fixed` infer at `0.000792 ms`).

| Workload | Stage | SwiftSci median (IQR), ms | Python median (IQR), ms | Faster engine | Factor |
|---|---|---:|---:|---|---:|
| `ts_hw_syn500_fixed` | fit | 0.010375 (0.000167) | 1.747167 (0.016250) | SwiftSci | 168.40× |
| `ts_hw_syn50k_fixed` | fit | 1.133667 (0.044167) | 135.822500 (1.004209) | SwiftSci | 119.81× |
| `ts_hw_syn50k_fixed` | infer | 0.049459 (0.005125) | 135.724125 (0.460542) | SwiftSci | 2744.17× |
| `ts_arima_syn500_matched_hr` | fit | 0.039750 (0.001166) | 3.222083 (0.134167) | SwiftSci | 81.06× |
| `ts_arima_syn500_matched_hr` | infer | 0.001333 (0.000084) | 0.307583 (0.007458) | SwiftSci | 230.74× |
| `ts_arima_dgp_syn50k_matched_hr` | fit | 2.460667 (0.071875) | 205.528125 (2.628708) | SwiftSci | 83.53× |
| `ts_arima_dgp_syn50k_matched_hr` | infer | 0.076958 (0.002208) | 0.336542 (0.036625) | SwiftSci | 4.37× |
| `ts_arima_dgp_syn500_matched_hr` | fit | 0.039000 (0.000709) | 3.298000 (0.039751) | SwiftSci | 84.56× |
| `ts_arima_dgp_syn500_matched_hr` | infer | 0.001292 (0.000500) | 0.307125 (0.007082) | SwiftSci | 237.71× |
| `ml_ols_regression` | fit | 0.443291 (0.006708) | 0.302083 (0.001167) | Python | 1.47× |
| `ml_ols_regression` | infer | 0.001375 (0.000125) | 0.032334 (0.005833) | SwiftSci | 23.52× |
| `ml_linear_svc` | fit | 12.347917 (0.311459) | 0.670500 (0.028166) | Python | 18.42× |
| `ml_linear_svc` | infer | 0.004375 (0.000291) | 0.095541 (0.017707) | SwiftSci | 21.84× |
| `ml_decision_tree_regressor` | fit | 1.847583 (0.019459) | 3.005750 (0.055167) | SwiftSci | 1.63× |
| `ml_decision_tree_regressor` | infer | 0.004208 (0.000125) | 0.039916 (0.001166) | SwiftSci | 9.49× |
| `ml_decision_tree_classifier` | fit | 4.040250 (0.317541) | 3.459250 (0.053625) | Python | 1.17× |
| `ml_decision_tree_classifier` | infer | 0.014000 (0.000458) | 0.095375 (0.002208) | SwiftSci | 6.81× |
| `ml_random_forest_regressor` | fit | 10.479292 (0.144042) | 110.870375 (0.479208) | SwiftSci | 10.58× |
| `ml_random_forest_regressor` | infer | 0.253500 (0.004625) | 1.703250 (0.054374) | SwiftSci | 6.72× |
| `ml_random_forest_classifier` | fit | 16.441708 (0.082124) | 53.053500 (1.324667) | SwiftSci | 3.23× |
| `ml_random_forest_classifier` | infer | 1.417625 (0.141167) | 3.619000 (0.047875) | SwiftSci | 2.55× |
| `ml_gbdt_regressor` | fit | 20.944875 (0.157125) | 101.510000 (1.367584) | SwiftSci | 4.85× |
| `ml_gbdt_regressor` | infer | 0.058083 (0.001417) | 0.203291 (0.004333) | SwiftSci | 3.50× |
| `ml_hist_gbdt_regressor` | fit | 12.999042 (0.125167) | 18.579542 (0.214709) | SwiftSci | 1.43× |
| `ml_hist_gbdt_regressor` | infer | 0.206000 (0.000666) | 0.421667 (0.005542) | SwiftSci | 2.05× |
| `ml_hist_gbdt_classifier` | fit | 15.147083 (0.051167) | 24.264709 (0.108584) | SwiftSci | 1.60× |
| `ml_hist_gbdt_classifier` | infer | 0.496041 (0.017584) | 1.288834 (0.021042) | SwiftSci | 2.60× |
| `cluster_pca` | fit | 0.146417 (0.001959) | 0.261834 (0.008542) | SwiftSci | 1.79× |
| `cluster_pca` | infer | 0.072750 (0.000792) | 0.059583 (0.002333) | Python | 1.22× |
| `cluster_kmeans` | fit | 0.204375 (0.004375) | 0.492708 (0.018125) | SwiftSci | 2.41× |
| `cluster_kmeans` | infer | 0.038166 (0.001584) | 0.072667 (0.001208) | SwiftSci | 1.90× |
| `stats_ddof1_100k` | fit | 1.141625 (0.010416) | 4.215166 (0.076542) | SwiftSci | 3.69× |
| `stats_ddof1_100k` | infer | 1.141625 (0.010416) | 4.215166 (0.076542) | SwiftSci | 3.69× |

Unmatched diagnostic disclosure: `ts_hw_syn500_opt` fit measured `0.276000 ms` (IQR `0.005291 ms`) for SwiftSci vs `48.660000 ms` (IQR `0.518333 ms`) for statsmodels; `ts_hw_syn500_fixed` infer measured `0.000792 ms` (IQR `0.000042 ms`, `< 1 µs` resolution-limited) vs `1.491958 ms` (IQR `0.016209 ms`).

### SwiftLLM KV-cache ablation

| Workload | Full forward (ms) | KV cache (ms) | Full forward / KV cache |
|---|---:|---:|---:|
| `llm_kvcache_ablation_p24_d16` | 6.371000 | 5.676208 | 1.12× |
| `llm_kvcache_ablation_p64_d24` | 9.710334 | 8.340917 | 1.16× |

## Interpretation and reproduction

Compare engines using the same input, output contract, estimator, and execution policy. `v3.14.0-release-gate/comparison` ran under `production-default` (AC power, native thread pools); `scientific_validation.json` ran under `legacy-capped` (AC power). Peak RSS in raw files includes runtime/library startup allocations.

Start at the [runner guide](../../README.md), review the [evidence index](README.md), and see the [deep accuracy audit](accuracy.deep.md) for acceptance and comparator scope.
