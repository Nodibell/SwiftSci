# Verified Performance Report — SwiftSci `v3.13.2-candidate` (Stage A)

- **Status:** `CANDIDATE` (Stage A — AC Power Verified, Uncommitted Working Tree)
- **Candidate Base Commit:** `0cf99f0470d0757d0a02bd3244c62d821ab144b0` (`v3.13.1`, `dirty: true`)
- **Stage A Source `tree_sha256`:** `5abd55baabb35be660ab990e97eb37bd009c9d8a955b09c447d290a083bdee30`
- **Final pre-release review source fingerprint:** `953e773629eabf731530b82edd6d74813e8d5f4b22bfec25b412ba9951eb4a9c` (four DocC pages and one whitespace-only edit in `ARIMA.swift`; benchmark executable inputs, scripts, and thresholds are unchanged apart from that source whitespace).
- **Machine / Memory / OS:** `Oleksiis-MacBook-Pro.local` (`Apple M4 Pro`, `arm64`, `24 GiB` / `25769803776 B`), `macOS-27.2.1-arm64-arm-64bit-Mach-O` (`26B131`)
- **Power State:** `AC -> AC` (`power_unchanged: true`)
- **Toolchain & Build:** `Apple Swift version 6.4.2 (swiftlang-6.4.2.1.9 clang-2100.4.1.1)`, `Xcode 27.2 (27B52)`, Release (`-O -whole-module-optimization`), `coverage_instrumentation: "absent"`, `MTL_FAST_MATH=NO`, `MTL_MATH_MODE=SAFE`, `MTL_MATH_FP32_FUNCTIONS=PRECISE`
- **Worker Binaries:**
  - Standardized sweeps & acceptance (`SwiftSciBenchmarkWorker`): `37d6108f6458fa813e7364926a7fa4fc673d42f57b6a7bdd29d815c724d6d4a5`
  - Scientific validation suite (`SwiftSciBenchmarkWorker`): `3e7e9e128f31849c9c524dd174ecc5ed18a2ebf39af3d8ee1cda840ef5389eba` (built from identical `Sources/` and `Benchmarks/Worker/` prior to `.build.json` refresh)
- **Python Environments:**
  - Standardized CLI suites (`Benchmarks/.venv-standardized`): `Python 3.14.6`, `NumPy 2.5.3`, `Pandas 3.0.6`, `SciPy 1.18.1`, `PyArrow 25.0.1`, `Polars 1.44.2`, `DuckDB 1.5.6`.
  - Scientific validation suite (`/Users/oleksiichumak/.pyenv/versions/3.11.9/bin/python3.11`): `Python 3.11.9`, `numpy 2.3.5`, `pandas 3.0.2`, `scipy 1.17.1`, `scikit-learn 1.8.0`, `statsmodels 0.14.6`, `shap 0.51.0` (`accelerate` BLAS/LAPACK).

> **Important Measurement Notes:**
> 1. All Peak RSS numbers in this report measure **whole-process `ru_maxrss`** (framework/runtime startup memory + workload allocations).
> 2. Stages with SwiftSci median `< 0.0010 ms` (`< 1000 ns`) are marked `<1us` / `RES<1us` and excluded from speedup ratios.
> 3. Cross-version comparisons against `v3.12.0` span an OS/toolchain change (`macOS 26B123 / Swift 6.4` -> `macOS 26B131 / Swift 6.4.2`).

---

## 1. Standardized Core Sweeps (`standard` & `standard-production-default`)

Both profiles passed `14/14` cases and received `certificate.json`: `"passed"`.

| Case | `legacy-capped` SwiftSci (ms) | `legacy-capped` Pandas (ms) | Speedup | `production-default` SwiftSci (ms) | `production-default` Pandas (ms) | Speedup | Peak RSS Swift / Pandas (`legacy`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `medium-csv-read` | `2.6113` | `10.4799` | **4.01x** | `2.2389` | `10.2691` | **4.59x** | `39.2 / 216.2 MiB` |
| `medium-filter` | `0.2217` | `0.7615` | **3.44x** | `0.2259` | `0.7625` | **3.38x** | `34.0 / 183.5 MiB` |
| `medium-sort` | `2.2446` | `2.9053` | **1.29x** | `2.2763` | `2.8533` | **1.25x** | `44.8 / 184.5 MiB` |
| `medium-group-sum` | `0.3987` | `0.5192` | **1.30x** | `0.3862` | `0.5293` | **1.37x** | `22.9 / 172.9 MiB` |
| `medium-flat-matrix` | `0.1113` | `0.6612` | **5.94x** | `0.1094` | `0.6754` | **6.17x** | `26.2 / 178.9 MiB` |
| `medium-target` | `0.0695` | `0.1016` | **1.46x** | `0.0564` | `0.1086` | **1.93x** | `24.8 / 172.9 MiB` |
| `medium-standard-scale` | `2.0309` | `0.8999` | `0.44x` (Pandas 2.26x) | `2.0082` | `0.9680` | `0.48x` (Pandas 2.07x) | `43.5 / 181.8 MiB` |
| `medium-minmax-scale` | `0.7927` | `0.8193` | **1.03x** | `0.7823` | `0.7876` | **1.01x** | `34.9 / 183.3 MiB` |
| `medium-mean` | `0.0075` | `0.0151` | **2.02x** | `0.0071` | `0.0144` | **2.03x** | `22.0 / 171.0 MiB` |
| `medium-variance` | `0.0525` | `0.0550` | **1.05x** | `0.0525` | `0.0555` | **1.06x** | `22.7 / 172.7 MiB` |
| `medium-stddev` | `0.0525` | `0.0559` | **1.07x** | `0.0543` | `0.0552` | **1.02x** | `22.7 / 171.1 MiB` |
| `nist-mean` | `0.0002` (`<1us`) | `0.0030` | `RES<1us` | `0.0002` (`<1us`) | `0.0031` | `RES<1us` | `10.5 / 152.4 MiB` |
| `nist-variance` | `0.0008` (`<1us`) | `0.0073` | `RES<1us` | `0.0007` (`<1us`) | `0.0073` | `RES<1us` | `10.6 / 152.8 MiB` |
| `nist-stddev` | `0.0008` (`<1us`) | `0.0080` | `RES<1us` | `0.0008` (`<1us`) | `0.0073` | `RES<1us` | `10.5 / 152.9 MiB` |

- `run_sha256`: `standard` = `3d06fbfc83a8266c`, `standard-production-default` = `3203c8328f8ce4c0`.

---

## 2. NIST Sweep (`nist`, `run_sha256: fda21eacd976a5ce`, `27/27` passed)

All 27 NIST cases passed audit and accuracy checks. Twenty-five cases run `< 0.0010 ms` (`0.0001–0.0008 ms`) on SwiftSci (`10.4–10.6 MiB` RSS vs Pandas `152.3–153.7 MiB`) and are suppressed (`RES<1us`). Only the two `pidigits` second-moment cases exceed `1 us`:
- `nist-pidigits-variance`: SwiftSci **`0.0029 ms`** (`10.8 MiB`) vs Pandas `0.0099 ms` (`152.8 MiB`), **3.39x**.
- `nist-pidigits-stddev`: SwiftSci **`0.0029 ms`** (`10.8 MiB`) vs Pandas `0.0102 ms` (`153.1 MiB`), **3.50x**.

---

## 3. Four-Engine Comparisons (`public-data` & `tabular-migration`)

### 3.1 `public-data` (`run_sha256: 3f7a7013cd1d8d1d`, `11/11` passed)

| Case | SwiftSci (ms) | Pandas (ms) | Polars (ms) | DuckDB (ms) | RSS (Swift / Pandas / Polars / DuckDB MiB) |
| --- | ---: | ---: | ---: | ---: | --- |
| `medium-uniform-h2o-q1` | `1.9874` | `1.2252` | **`0.8971`** | `0.9126` | `40.3 / 179.1 / 159.6 / 149.5` |
| `medium-uniform-h2o-q2` | `7.0143` | `2.9897` | `3.1681` | **`2.1131`** | `41.2 / 195.4 / 171.5 / 154.6` |
| `medium-uniform-h2o-q3` | `2.5073` | `1.8515` | `1.6304` | **`1.2050`** | `40.4 / 180.2 / 161.5 / 151.8` |
| `medium-uniform-h2o-q4` | `0.6142` | `1.1346` | `1.9508` | **`0.4350`** | `39.1 / 182.0 / 157.4 / 146.3` |
| `medium-uniform-h2o-q5` | `0.8437` | `1.1881` | `1.1098` | **`0.5023`** | `40.4 / 179.7 / 157.9 / 147.2` |
| `medium-hot-key-h2o-q1` | `1.7929` | `1.1649` | **`0.8443`** | `1.0527` | `38.0 / 180.6 / 159.6 / 148.6` |
| `medium-hot-key-h2o-q2` | `5.9581` | `2.8063` | `2.3948` | **`2.1073`** | `39.0 / 189.0 / 170.6 / 151.2` |
| `medium-hot-key-h2o-q3` | `2.3828` | `2.0340` | `1.6806` | **`1.5642`** | `38.3 / 181.3 / 162.1 / 149.5` |
| `medium-hot-key-h2o-q4` | `1.1483` | `2.2433` | `1.9417` | **`0.9732`** | `38.1 / 183.5 / 157.9 / 145.0` |
| `medium-hot-key-h2o-q5` | `1.3781` | `1.7630` | `1.1688` | **`1.0092`** | `39.3 / 179.7 / 158.2 / 145.3` |
| `wine-red-pipeline` | **`0.3650`** | `1.5242` | `0.9532` | `18.6392` | `13.7 / 156.9 / 89.3 / 95.8` |

### 3.2 `tabular-migration` (`run_sha256: 9436c2c5a50b545a`, `19/19` passed)

| Case | SwiftSci (ms) | Pandas (ms) | Polars (ms) | DuckDB (ms) | Winner |
| --- | ---: | ---: | ---: | ---: | --- |
| `medium-group-sum-mean` | `0.5056` | `1.6587` | `1.3113` | **`0.3322`** | DuckDB |
| `medium-row-sum` | **`3.8082`** | `18.9698` | `23.4544` | `17.2836` | **SwiftSci** |
| `medium-inner-join` | `2.0407` | **`0.8654`** | `2.4440` | `2.1318` | Pandas |
| `medium-pearson` | `0.1543` | `0.2567` | **`0.0727`** | `0.4728` | Polars |
| `medium-spearman` | `3.0222` | `3.3427` | **`2.3875`** | `16.9570` | Polars |
| `mixed-medium-csv-read` | `8.7245` | `14.0006` | **`6.3258`** | `36.0847` | Polars |
| `mixed-medium-filter` | **`0.2958`** | `0.8630` | `0.3847` | `1.4817` | **SwiftSci** |
| `mixed-medium-group-sum-mean` | `2.2623` | `2.0980` | **`0.7815`** | `0.9847` | Polars |
| `mixed-medium-sort` | `2.3292` | `3.2017` | **`1.4228`** | `7.0698` | Polars |
| `mixed-medium-flat-matrix` | **`0.1082`** | `0.6635` | `0.7948` | `1.4659` | **SwiftSci** |
| `mixed-medium-row-sum` | **`3.8965`** | `47.7095` | `26.3105` | `24.7872` | **SwiftSci** |
| `mixed-medium-inner-join` | `2.1265` | **`0.9421`** | `3.1008` | `2.8680` | Pandas |
| `mixed-medium-parquet-write` | `7.7994` | `8.5417` | **`3.8057`** | `4.1450` | Polars |
| `mixed-medium-standard-scale` | `2.0072` | **`0.9334`** | `1.3662` | `3.1952` | Pandas |
| `mixed-medium-minmax-scale` | **`0.7597`** | `0.7741` | `1.0756` | `3.4134` | **SwiftSci** |
| `parquet-medium-read` | `23.3771` | `2.5157` | **`1.3910`** | `2.5570` | Polars |
| `large-mean` | **`0.0809`** | `0.1178` | `0.1015` | `0.6867` | **SwiftSci** |
| `large-variance` | `0.5270` | `0.4767` | **`0.4463`** | `3.8136` | Polars |
| `large-stddev` | `0.5351` | `0.4807` | **`0.4429`** | `3.8965` | Polars |

- Summary wins (`tabular-migration`): Polars `9`, SwiftSci `6`, Pandas `3`, DuckDB `1`.

---

## 4. `migration` Suite (`run_sha256: 16e3a37b7686e791`, `36/36` passed)

Resolved cases (`>= 1 us`): `35` (`tiny-rag-summary` is `0.0005 ms` `<1us` vs Pandas `0.0169 ms`). SwiftSci beats Pandas on **25/35** resolved cases.

| Case | SwiftSci (ms) | Pandas (ms) | Ratio (Pandas / Swift) | Peak RSS (Swift / Pandas) |
| --- | ---: | ---: | ---: | --- |
| `medium-csv-stream-read` | `2.1937` | `13.2273` | **6.03x** | `36.0 / 242.3 MiB` |
| `medium-csv-stream-filter` | `2.6392` | `14.3227` | **5.43x** | `30.6 / 235.2 MiB` |
| `medium-csv-stream-group` | `3.1240` | `25.8669` | **8.28x** | `19.4 / 229.2 MiB` |
| `medium-group-sum-mean` | `0.5258` | `1.6973` | **3.23x** | `23.6 / 173.9 MiB` |
| `medium-row-sum` | `4.0524` | `19.2840` | **4.76x** | `22.0 / 171.7 MiB` |
| `medium-inner-join` | `2.0825` | `0.8466` | `0.41x` (Pandas 2.46x) | `55.2 / 187.3 MiB` |
| `medium-pearson` | `0.1499` | `0.2537` | **1.69x** | `24.0 / 176.8 MiB` |
| `medium-spearman` | `3.0191` | `3.5310` | **1.17x** | `30.5 / 175.3 MiB` |
| `mixed-medium-csv-read` | `8.1422` | `14.2640` | **1.75x** | `45.2 / 255.4 MiB` |
| `mixed-medium-csv-stream-read` | `8.2110` | `17.6197` | **2.15x** | `43.8 / 236.0 MiB` |
| `mixed-medium-csv-stream-filter` | `9.2276` | `19.2662` | **2.09x** | `35.6 / 232.2 MiB` |
| `mixed-medium-csv-stream-group` | `10.2604` | `30.6601` | **2.99x** | `20.5 / 216.8 MiB` |
| `mixed-medium-filter` | `0.3072` | `0.8828` | **2.87x** | `43.5 / 196.0 MiB` |
| `mixed-medium-group-sum-mean` | `2.3399` | `2.1720` | `0.93x` (Pandas 1.08x) | `29.0 / 176.9 MiB` |
| `mixed-medium-sort` | `2.2060` | `3.1690` | **1.44x** | `66.2 / 211.7 MiB` |
| `mixed-medium-flat-matrix` | `0.1184` | `0.6637` | **5.61x** | `31.3 / 181.7 MiB` |
| `mixed-medium-row-sum` | `3.8742` | `47.4630` | **12.25x** | `29.0 / 175.1 MiB` |
| `mixed-medium-inner-join` | `2.1056` | `0.8696` | `0.41x` (Pandas 2.42x) | `68.8 / 207.8 MiB` |
| `mixed-medium-parquet-write` | `7.8106` | `8.5948` | **1.10x** | `72.8 / 244.6 MiB` |
| `mixed-medium-standard-scale` | `1.9449` | `0.8774` | `0.45x` (Pandas 2.22x) | `35.3 / 184.1 MiB` |
| `mixed-medium-minmax-scale` | `0.7550` | `0.7928` | **1.05x** | `41.4 / 184.3 MiB` |
| `parquet-medium-read` | `23.5250` | `2.5375` | `0.11x` (Pandas 9.27x) | `43.3 / 206.4 MiB` |
| `medium-welch` | `0.2950` | `0.6730` | **2.28x** | `22.7 / 171.4 MiB` |
| `medium-regression-metrics` | `0.8756` | `0.3236` | `0.37x` (Pandas 2.71x) | `23.2 / 173.3 MiB` |
| `medium-roc-auc` | `2.1491` | `2.0537` | `0.96x` (Pandas 1.05x) | `26.0 / 177.6 MiB` |
| `medium-onehot` | `15.3531` | `3.7142` | `0.24x` (Pandas 4.13x) | `77.0 / 226.7 MiB` |
| `medium-tfidf` | `100.1414` | `113.4361` | **1.13x** | `54.4 / 233.5 MiB` |
| `tiny-sqlite-ingest` | `0.0495` | `0.1164` | **2.35x** | `11.8 / 155.9 MiB` |
| `tiny-rag-summary` | `0.0005` (`<1us`) | `0.0169` | `RES<1us` | `11.0 / 154.1 MiB` |
| `tiny-pool-dice` | `0.0033` | `0.0085` | **2.58x** | `11.0 / 155.0 MiB` |
| `large-mean` | `0.0784` | `0.1186` | **1.51x** | `118.8 / 251.1 MiB` |
| `large-variance` | `0.5281` | `0.4871` | `0.92x` (Pandas 1.08x) | `118.8 / 258.5 MiB` |
| `large-stddev` | `0.5250` | `0.4765` | `0.91x` (Pandas 1.10x) | `118.8 / 259.3 MiB` |
| `medium-student` | `0.2930` | `0.7046` | **2.40x** | `21.9 / 172.0 MiB` |
| `medium-paired` | `0.2793` | `0.5584` | **2.00x** | `23.2 / 172.6 MiB` |
| `medium-anova` | `0.4356` | `0.8472` | **1.94x** | `22.5 / 174.2 MiB` |

---

## 5. Scientific Validation Suite Timings (`scientific_validation.json`, `39/39` Passed)

All `5`-process × `5`-sample (+ `2`-warmup) runs; sub-`1 us` stages excluded (`5` total resolution-limited stages). Non-comparable estimators / internal ablations never appear as Python speedups.

### 5.1 Comparable Estimator / Mathematical Speedups

| Workload | Fit Swift / Py (ms) | Fit Speedup | Infer Swift / Py (ms) | Infer Speedup | Peak RSS (Swift / Py MiB) |
| --- | --- | --- | --- | --- | --- |
| `ts_hw_syn500_fixed` | `0.010709 / 1.829625` | **SwiftSci 170.85x** | `0.000833 / 1.525125` | `RES<1us` | `10.1 / 298.8` |
| `ts_hw_syn50k_fixed` | `1.144542 / 138.079834` | **SwiftSci 120.64x** | `0.048000 / 139.002667` | **SwiftSci 2895.89x** | `13.8 / 310.0` |
| `ts_arima_syn500_matched_hr` | `0.039459 / 3.473625` | **SwiftSci 88.03x** | `0.001209 / 0.343375` | **SwiftSci 284.02x** | `10.4 / 299.2` |
| `ts_arima_dgp_syn500_matched_hr` | `0.039208 / 3.397833` | **SwiftSci 86.66x** | `0.001250 / 0.346250` | **SwiftSci 277.00x** | `10.4 / 299.1` |
| `ts_arima_dgp_syn50k_matched_hr` | `2.515708 / 210.733083` | **SwiftSci 83.77x** | `0.077334 / 0.377458` | **SwiftSci 4.88x** | `19.8 / 617.8` |
| `ml_ols_regression` | `0.451958 / 0.260541` | Python **1.73x** (`0.58x`) | `0.001458 / 0.023542` | **SwiftSci 16.15x** | `25.9 / 298.8` |
| `ml_linear_svc` | `12.238166 / 0.621083` | Python **19.70x** (`0.05x`) | `0.002458 / 0.059417` | **SwiftSci 24.17x** | `26.2 / 299.0` |
| `ml_decision_tree_regressor` | `1.839792 / 3.503209` | **SwiftSci 1.90x** | `0.004166 / 0.025000` | **SwiftSci 6.00x** | `12.7 / 298.8` |
| `ml_decision_tree_classifier` | `3.897416 / 3.923000` | **SwiftSci 1.01x** | `0.012458 / 0.072167` | **SwiftSci 5.79x** | `13.2 / 299.1` |
| `ml_random_forest_regressor` | `10.462792 / 130.841709` | **SwiftSci 12.51x** | `0.217959 / 1.850250` | **SwiftSci 8.49x** | `18.8 / 299.7` |
| `ml_random_forest_classifier` | `15.746666 / 58.421792` | **SwiftSci 3.71x** | `1.072958 / 3.833500` | **SwiftSci 3.57x** | `19.8 / 299.3` |
| `ml_gbdt_regressor` | `20.416209 / 115.643292` | **SwiftSci 5.66x** | `0.055292 / 0.194208` | **SwiftSci 3.51x** | `13.1 / 299.0` |
| `ml_hist_gbdt_regressor` | `13.143250 / 18.624167` | **SwiftSci 1.42x** | `0.194625 / 0.415125` | **SwiftSci 2.13x** | `12.8 / 299.0` |
| `ml_hist_gbdt_classifier` | `14.930625 / 24.617875` | **SwiftSci 1.65x** | `0.476500 / 1.234791` | **SwiftSci 2.59x** | `13.4 / 299.3` |
| `cluster_pca` | `0.147125 / 0.246875` | **SwiftSci 1.68x** | `0.067291 / 0.043291` | Python **1.55x** (`0.64x`) | `11.8 / 299.1` |
| `cluster_kmeans` | `0.202875 / 0.513084` | **SwiftSci 2.53x** | `0.038208 / 0.058416` | **SwiftSci 1.53x** | `11.5 / 298.7` |
| `stats_ddof1_100k` | `1.155209 / 1.181541` | **SwiftSci 1.02x** | `1.155209 / 1.181541` | **SwiftSci 1.02x** | `18.2 / 300.8` |

### 5.2 Non-Comparable, Expected-Error, & Internal SwiftSci Workloads

- **Non-comparable Python baselines (`NON_COMPARABLE_ESTIMATOR_OR_WEIGHTS`):** `ts_hw_syn500_opt` (`0.2674` vs `49.0127 ms`), `ts_arima_syn500_vs_statespace` (`0.0390` vs `8.8798 ms`), `ts_arima_syn50k_vs_statespace` (`2.6004` vs `494.4663 ms`), `ts_sarima_airpassengers_110_010` (`0.0082` vs `6.2164 ms`), `ts_sarima_airpassengers_011_011` (`0.0263` vs `26.0171 ms`), `ml_logistic_regression` (`4.4978` vs `0.7731 ms`), `explain_treeshap_dt_k4` (`0.0225` vs `0.0712 ms`), `explain_treeshap_rf_k8` (`0.5660` vs `0.1758 ms`), `explain_treeshap_dt_k16_mc` (`141.9783` vs `0.1308 ms`), `stats_ddof1_small5` (`0.000292 ms` `<1us` vs `0.5235 ms`).
- **Expected-error fail-closed workloads (`EXCLUDED_EXPECTED_ERROR`):** `ts_sarima_expect_fail_syn500` & `ts_sarima_expect_fail_syn50k` (`SwiftMLError`, `is_fitted=false`).
- **Swift-only controls & diagnostics:** `ts_sarima_airpassengers_144_g037` (`0.0094 ms` fit, `0.00075 ms` `<1us` infer), `ts_auto_arima_valid` (`0.2275 ms`), `ts_auto_arima_zero_variance` (`0.0063 ms`), `ts_auto_arima_poly_sign_check` (`0.0081 ms`), `ts_auto_arima_poly_unstable_check` (`0.0070 ms`), `anomaly_syn144_s04` (`0.0112 ms`), `anomaly_syn500_holdout` (`0.0487 ms`), `anomaly_syn500_trended_diagnostic` (`0.0342 ms`).
- **SwiftLLM KV-cache ablation (`swift_only: true`):**
  - `llm_kvcache_ablation_p24_d16`: `7.4457 ms` (full-forward) -> `6.4525 ms` (KV-cache), **1.15x** (`max_abs_logit_diff = 3.58e-07`, `34.3 MiB` RSS).
  - `llm_kvcache_ablation_p64_d24`: `10.6213 ms` (full-forward) -> `9.2183 ms` (KV-cache), **1.15x** (`max_abs_logit_diff = 4.77e-07`, `34.6 MiB` RSS).
