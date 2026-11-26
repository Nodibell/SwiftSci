# SwiftSci 3.13.1 Deep Performance Audit

> **Navigation**: [← v3.13.1 Evidence Index](README.md) · [Performance Overview](../../../PERFORMANCE/3.13.1.md) · [Deep Accuracy Audit](accuracy.deep.md) · [Latest (`3.13.2`)](../v3.13.2/performance.deep.md)
>
> **Status — Historical Snapshot (`v3.13.1`, Superseded by `v3.13.2`)**: Evaluates the `3.13.1` working tree (`HEAD: 8e45a1e88ca0f8ae072321fe7852d81aec6a6fe6`, generated `2026-11-14T20:12:12Z`) prior to the `v3.13.2` correctness patch and retirement of legacy `--research` harnesses.

> [!WARNING]
> **v3.13.2 Benchmark Cleanup Notice**: Legacy `--research` harnesses (`SwiftSciBenchmarks --research`, `Benchmarks/Python/benchmarks.py`, `Benchmarks/CSVAcceleration/`) referenced below were retired and removed in `v3.13.2`. For published AC-power benchmarks and scientific validation results, see [`PERFORMANCE/3.13.2.md`](../../../PERFORMANCE/3.13.2.md) and [`v3.13.2/README.md`](../v3.13.2/README.md).

---

## Workload Provenance & Audit Clarifications

### 1. Environment & Platform Provenance Across Releases

| Release / Run | Date | Hardware & Power | macOS & Build | Swift / Xcode Toolchain | Python Controller & Worker |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **`3.11.0`** (`standard-3.11.0`) | `2026-10-14` | Apple M4 Pro (`arm64`, 24 GiB, AC) | `macOS 27.0.1` (`arm-64bit-Mach-O`) | Swift `6.4` (`swiftlang-6.4.0.34.1`), Xcode `27.0 (27A266a)` | Controller & worker: Python `3.14.6` (`Pandas 3.0.6`, `NumPy 2.5.3`, `SciPy 1.18.1`, `PyArrow 25.0.1`) |
| **`3.12.0`** (`standard-3.12.0`, `migration-3.12.0`) | `2026-10-28` | Apple M4 Pro (`arm64`, 24 GiB, AC) | `macOS 27.2 (26B123)` | Swift `6.4` (`swiftlang-6.4.0.34.1`), Xcode `27.1 (27A9275)` | Controller & worker: Python `3.14.6` (`Pandas 3.0.6`, `NumPy 2.5.3`, `SciPy 1.18.1`, `PyArrow 25.0.1`) |
| **`3.13.0`** | — | **No independent classical run** | — | — | Classical/tabular metrics carried forward from `3.12.0` (`SwiftLLM` / `SwiftNLP` only changed) |
| **`3.13.1`** (`standard-3.13.1`, `migration-3.13.1`) | `2026-11-14` | Apple M4 Pro (`arm64`, 24 GiB, AC, `MTL_FAST_MATH=NO`, `MTL_MATH_MODE=SAFE`) | `macOS 27.2.1 (26B131)` | Swift `6.4.2` (`swiftlang-6.4.2.1.9`), Xcode `27.2 (27B52)`, Release (`-O -whole-module-optimization`) | `standard-3.13.1`: controller Python `3.11.9` / Pandas worker Python `3.14.6`; `migration-3.13.1` & `comparison-suite-20261114-2033`: both Python `3.14.6` (`Pandas 3.0.6`, `NumPy 2.5.3`, `SciPy 1.18.1`, `PyArrow 25.0.1`). Extended `--research` / patched Python stack: Python `3.11.9` (`Scikit-Learn 1.8.0`, `Statsmodels 0.14.6`, `SHAP 0.51.0`, `PyTorch 2.11.0`, `NLTK 3.10.3`). |

### 2. Investigation of Flagged Cross-Version Slowdowns (`3.12.0` → `3.13.1`)

Compared against `2026-10-28` (`standard-3.12.0` / `migration-3.12.0` on `macOS 27.2`, Swift `6.4`), `work/Runs/*-3.13.1` (`2026-11-14` on `macOS 27.2.1`, Swift `6.4.2`) flagged `2/14` `standard` regressions (`medium-sort +10.4%`, `medium-minmax-scale +12.2%`) and `14/36` `migration` regressions (`+5.0%` to `+16.9%`). Controlled inspection confirmed this is **OS/toolchain/run-level variance rather than source-code regressions**:
1. **Unmodified Python/Pandas Stack Slowed Too**: On the exact same machine with unchanged package versions (`Pandas 3.0.6 / NumPy 2.5.3 / SciPy 1.18.1 / PyArrow 25.0.1 / Python 3.14.6`), Python showed `0` improvements and `5/14` regressions in `standard`, plus `0` improvements and `19/36` regressions in `migration`.
2. **Zero Diff in Flagged Swift Kernels**: `git diff` across `Sources/` confirmed zero algorithmic edits in `sort`, `MinMaxScaler`, `regressionMetrics`, `oneWayANOVA`, `GroupedDataFrame.sum`, or `Stats.variance` between `3.12.0` and `3.13.1`.
3. **Clean Follow-Up Standardized Run (`comparison-suite-20261114-2033/standard`)**: Executing the 14-case `standard` profile under `.venv-standardized` (`Python 3.14.6`) measured `medium-csv-read = 2.3662 ms`, `medium-filter = 0.2394 ms`, `medium-sort = 2.2702 ms`, `medium-group-sum = 0.3896 ms`, `medium-flat-matrix = 0.1157 ms`, `medium-target = 0.0576 ms`, `medium-standard-scale = 2.0342 ms`, `medium-minmax-scale = 0.7915 ms`.

### 3. CSV Workloads: `csv-read-v2` vs `100k_NullTokens` Round-Trip

- **Standardized `medium-csv-read`** ([`Benchmarks/Specs/workloads/csv-read-v2.json`](../../Specs/workloads/csv-read-v2.json)): `100,000 × 5` **valid finite non-null rows** with a **complete explicit schema** (`id/group Int64`, `x/y Float64`; no null tokens, no type inference, no quoting). `standard-3.13.1` measured **`2.798 ms`** (`2.366 ms` in `comparison-suite-20261114-2033`), `39.4 MB` RSS vs Pandas `11.474 ms` (`215.9 MB` RSS) (**4.10× faster**).
- **Quoted Null-Token CSV Read/Write (`CSV_*_100k_NullTokens`)**: Explicitly tests `"NA"`, `"null"`, `"NaN"`, `"None"`, `""`, and true `nil` over `100,000 × 3` rows (`CSV_Read_Eager_100k_NullTokens = 3.512 ms` vs Pandas `10.618 ms`, **3.02×**; `CSV_Write_100k_NullTokens = 15.714 ms` vs Pandas `57.223 ms`, **3.64×**). SwiftSci's `2,290,455` output bytes vs Pandas `2,165,456` bytes differ by `+124,999 B` (`2 quote bytes × 5/8 × 100,000 = +125,000 B` protecting null-token/empty literals, minus `1` trailing EOF `\n`), preventing Pandas's default `read_csv` collapse of `75,000` cells into `NaN`.

### 4. Hash Join Workload Traceability Across Profiles

Do not merge Hash Join latencies across incompatible harnesses:

| Benchmark Profile | Join Operation / Schema | SwiftSci Baseline | SwiftSci 3.13.1 | Python (Pandas) | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Historical Research (`3.11.0`)** | Legacy `--research` in-process microbenchmark (`100k` rows) | `2.701 ms` (`3.11.0`) | — | `9.800 ms` (`3.11.0` table) | Legacy harness prior to `inner-join-v1` |
| **Standardized `migration`** (`inner-join-v1`, `bench.py`) | Process-isolated `.inner` (`100k` rows, prepared right table, `Std` / `Mixed`) | `10.635 / 13.808 ms` (`3.12.0`) | **`2.185 / 2.209 ms`** | `0.9241 / 0.9430 ms` | **-79.5% / -84.0% vs `3.12.0`** (`z=113.33 / 29.78`) |
| **Unvalidated `--research` (`3.13.1`)** | In-process `DataFrameBenchmarks.swift` (`100k` rows) | — | **`2.183 ms` med (`2.240 ms` mean)** | `0.4249 ms` med | Separate non-standardized in-process harness |

---

## 1. Standardized Core (`standard`, 14 cases × 3 batches × 5 samples, 2 warmups)

- **SwiftSci 3.13.1 (`standard-3.13.1`) vs `3.12.0` (`standard-3.12.0`)**: 0 improvements, 2 regressions, 12 inconclusive / within noise.
- **Python/Pandas (`3.13.1` run vs `3.12.0` run)**: 0 faster, 5 slower, 9 inconclusive.

| Case ID | Sw 3.12.0 Med (ms) | Sw 3.13.1 Med (ms) | Sw 3.13.1 Mean ± SD (ms) | Sw CV% / IQR (ms) | Δ vs 3.12.0 | Verdict | Py 3.13.1 Med (ms) | Sw vs Py | Sw RSS (MB) |
| :--- | ---: | ---: | ---: | ---: | :---: | :--- | ---: | ---: | ---: |
| `medium-csv-read` | 2.203 | 2.798 | 2.798 ± 0.2530 | 9.0% / 0.3067 | +27.0% | inconclusive (overlapping batches; `2.366 ms` in `2033`) | 11.474 | **4.10×** | 39.4 |
| `medium-filter` | 0.2284 | 0.2378 | 0.2543 ± 0.0296 | 11.6% / 0.0344 | +4.1% | inconclusive (`< 0.025 ms` floor) | 0.8292 | **3.49×** | 34.8 |
| `medium-sort` | 2.140 | 2.362 | 2.396 ± 0.1095 | 4.6% / 0.1582 | +10.4% | regression (disjoint batches `z=6.25`; `2.270 ms` in `2033`) | 2.963 | **1.25×** | 46.2 |
| `medium-group-sum` | 0.3799 | 0.4043 | 0.4022 ± 0.0257 | 6.4% / 0.0252 | +6.4% | inconclusive (`< 0.025 ms` floor; `0.390 ms` in `2033`) | 0.5411 | **1.34×** | 23.5 |
| `medium-flat-matrix` | 0.1122 | 0.1218 | 0.1341 ± 0.0321 | 23.9% / 0.0188 | +8.6% | inconclusive (`< 0.025 ms` floor) | 0.7472 | **6.13×** | 30.0 |
| `medium-target` | 0.0566 | 0.0575 | 0.0591 ± 0.0050 | 8.4% / 0.0030 | +1.6% | inconclusive (`< 0.025 ms` floor) | 0.1076 | **1.87×** | 25.5 |
| `medium-standard-scale` | 1.966 | 2.106 | 2.165 ± 0.1684 | 7.8% / 0.2354 | +7.1% | inconclusive (overlapping batches) | 1.111 | **0.53×** | 40.2 |
| `medium-minmax-scale` | 0.7362 | 0.8260 | 0.8603 ± 0.0794 | 9.2% / 0.0875 | +12.2% | regression (`z=2.69`; `0.792 ms` in `2033`) | 1.007 | **1.22×** | 37.1 |
| `medium-mean` | 0.0076 | 0.0079 | 0.0078 ± 0.0003 | 3.3% / 0.0004 | +3.3% | inconclusive (`< 0.025 ms` floor) | 0.0153 | **1.94×** | 22.6 |
| `medium-variance` | 0.0532 | 0.0543 | 0.0548 ± 0.0021 | 3.9% / 0.0002 | +2.1% | inconclusive (`< 0.025 ms` floor) | 0.0555 | **1.02×** | 22.5 |
| `medium-stddev` | 0.0517 | 0.0534 | 0.0538 ± 0.0011 | 2.0% / 0.0014 | +3.2% | inconclusive (`< 0.025 ms` floor) | 0.0565 | **1.06×** | 21.9 |
| `nist-mean` | 0.0002 | 0.0002 | 0.0002 ± 0.0001 | 30.2% / 0.0001 | -0.5% | inconclusive (`< 0.025 ms` floor) | 0.0035 | **16.62×** | 10.5 |
| `nist-variance` | 0.0008 | 0.0008 | 0.0008 ± 0.0001 | 7.8% / 0.0001 | -5.0% | inconclusive (`< 0.025 ms` floor) | 0.0076 | **9.58×** | 10.5 |
| `nist-stddev` | 0.0008 | 0.0008 | 0.0008 ± 0.0000 | 6.3% / 0.0001 | -5.2% | inconclusive (`< 0.025 ms` floor) | 0.0083 | **10.48×** | 10.5 |

---

## 2. Standardized Migration (`migration`, 36 cases × 3 batches × 5 samples, 2 warmups)

- **SwiftSci 3.13.1 (`migration-3.13.1`) vs `3.12.0` (`migration-3.12.0`)**: **3 improvements** (`medium-inner-join -79.5%`, `mixed-medium-filter -27.9%`, `mixed-medium-inner-join -84.0%`), 14 regressions, 19 inconclusive / within noise.
- **Python/Pandas (`3.13.1` run vs `3.12.0` run)**: 0 faster, 19 slower, 17 inconclusive.

| Case ID | Sw 3.12.0 Med (ms) | Sw 3.13.1 Med (ms) | Sw 3.13.1 Mean ± SD (ms) | Sw CV% / IQR (ms) | Δ vs 3.12.0 | Verdict | Py 3.13.1 Med (ms) | Sw vs Py | Sw RSS (MB) |
| :--- | ---: | ---: | ---: | ---: | :---: | :--- | ---: | ---: | ---: |
| `medium-csv-stream-read` | 2.206 | 2.441 | 2.438 ± 0.2319 | 9.5% / 0.1468 | +10.6% | regression (`z=2.39`) | 13.379 | **5.48×** | 40.0 |
| `medium-csv-stream-filter` | 2.578 | 2.879 | 2.905 ± 0.1005 | 3.5% / 0.1041 | +11.7% | regression (`z=5.06`) | 15.594 | **5.42×** | 32.1 |
| `medium-csv-stream-group` | 2.935 | 3.180 | 3.246 ± 0.3172 | 9.8% / 0.1934 | +8.4% | inconclusive | 27.287 | **8.58×** | 18.8 |
| `medium-group-sum-mean` | 0.4851 | 0.5143 | 0.5139 ± 0.0168 | 3.3% / 0.0214 | +6.0% | inconclusive | 1.759 | **3.42×** | 23.4 |
| `medium-row-sum` | 3.850 | 4.015 | 4.018 ± 0.1559 | 3.9% / 0.2361 | +4.3% | inconclusive (`z=1.11`) | 19.357 | **4.82×** | 22.5 |
| `medium-inner-join` | 10.635 | 2.185 | 2.167 ± 0.0791 | 3.7% / 0.0819 | **-79.5%** | **improvement (`z=113.33`)** | 0.9241 | **0.42×** | 52.6 |
| `medium-pearson` | 0.1478 | 0.1694 | 0.1702 ± 0.0111 | 6.5% / 0.0073 | +14.6% | inconclusive (`< 0.025 ms` floor) | 0.2751 | **1.62×** | 23.9 |
| `medium-spearman` | 2.904 | 3.079 | 3.086 ± 0.0877 | 2.8% / 0.1238 | +6.0% | regression (`z=3.76`) | 3.947 | **1.28×** | 30.3 |
| `mixed-medium-csv-read` | 7.557 | 8.450 | 8.519 ± 0.2285 | 2.7% / 0.2855 | +11.8% | regression (`z=2.48`) | 15.105 | **1.79×** | 50.2 |
| `mixed-medium-csv-stream-read` | 7.868 | 8.356 | 8.351 ± 0.1155 | 1.4% / 0.1308 | +6.2% | regression (`z=3.47`) | 17.914 | **2.14×** | 47.4 |
| `mixed-medium-csv-stream-filter` | 8.638 | 9.334 | 9.783 ± 0.8639 | 8.8% / 1.612 | +8.1% | inconclusive (`z=0.65`) | 21.050 | **2.26×** | 37.2 |
| `mixed-medium-csv-stream-group` | 9.956 | 11.640 | 11.543 ± 0.3468 | 3.0% / 0.4824 | +16.9% | regression (`z=3.52`) | 30.552 | **2.62×** | 21.0 |
| `mixed-medium-filter` | 0.4138 | 0.2984 | 0.3005 ± 0.0208 | 6.9% / 0.0232 | **-27.9%** | **improvement (`z=15.98`)** | 0.8857 | **2.97×** | 42.2 |
| `mixed-medium-group-sum-mean` | 2.234 | 2.322 | 2.319 ± 0.0635 | 2.7% / 0.0444 | +3.9% | inconclusive (`z=2.54`) | 2.377 | **1.02×** | 28.9 |
| `mixed-medium-sort` | 2.484 | 2.416 | 2.394 ± 0.0846 | 3.5% / 0.1013 | -2.7% | inconclusive (`z=0.93`) | 3.401 | **1.41×** | 61.9 |
| `mixed-medium-flat-matrix` | 0.0995 | 0.1180 | 0.1276 ± 0.0249 | 19.5% / 0.0221 | +18.5% | inconclusive (`< 0.025 ms` floor) | 0.7172 | **6.08×** | 34.2 |
| `mixed-medium-row-sum` | 3.949 | 4.101 | 4.091 ± 0.1536 | 3.8% / 0.1121 | +3.9% | inconclusive (`z=2.03`) | 49.332 | **12.03×** | 29.0 |
| `mixed-medium-inner-join` | 13.808 | 2.209 | 2.223 ± 0.0874 | 3.9% / 0.0943 | **-84.0%** | **improvement (`z=29.78`)** | 0.9430 | **0.43×** | 63.7 |
| `mixed-medium-parquet-write` | 7.602 | 8.278 | 8.428 ± 0.4664 | 5.5% / 0.2368 | +8.9% | regression (`z=4.75`) | 8.208 | **0.99×** | 64.5 |
| `mixed-medium-standard-scale` | 1.917 | 2.036 | 2.100 ± 0.1209 | 5.8% / 0.1191 | +6.2% | inconclusive (`z=1.70`) | 1.005 | **0.49×** | 45.9 |
| `mixed-medium-minmax-scale` | 0.7173 | 0.7953 | 0.8361 ± 0.0760 | 9.1% / 0.1156 | +10.9% | inconclusive (`z=1.43`) | 0.8382 | **1.05×** | 41.9 |
| `parquet-medium-read` | 22.425 | 23.672 | 23.641 ± 0.2891 | 1.2% / 0.4090 | +5.6% | regression (`z=2.43`) | 2.728 | **0.12×** | 54.6 |
| `medium-welch` | 0.2845 | 0.3105 | 0.3319 ± 0.0596 | 18.0% / 0.0043 | +9.1% | regression (`z=2.34`) | 0.7329 | **2.36×** | 22.5 |
| `medium-regression-metrics` | 0.8194 | 0.9236 | 0.9256 ± 0.0184 | 2.0% / 0.0090 | +12.7% | regression (`z=12.16`) | 0.3448 | **0.37×** | 23.2 |
| `medium-roc-auc` | 2.115 | 2.563 | 3.766 ± 3.844 | 102.1% / 0.6754 | +21.1% | inconclusive (`z=0.37`) | 2.392 | **0.93×** | 29.2 |
| `medium-onehot` | 14.673 | 16.493 | 16.504 ± 0.7121 | 4.3% / 0.6987 | +12.4% | regression (`z=3.45`) | 3.692 | **0.22×** | 77.9 |
| `medium-tfidf` | 96.913 | 101.8 | 102.5 ± 1.821 | 1.8% / 0.6591 | +5.0% | regression (`z=3.42`) | 122.5 | **1.20×** | 57.4 |
| `tiny-sqlite-ingest` | 0.0541 | 0.0678 | 0.0984 ± 0.0435 | 44.2% / 0.0554 | +25.2% | inconclusive (`< 0.025 ms` floor) | 0.1160 | **1.71×** | 11.9 |
| `tiny-rag-summary` | 0.0007 | 0.0005 | 0.0010 ± 0.0009 | 86.5% / 0.0009 | -18.9% | inconclusive (`< 0.025 ms` floor) | 0.0178 | **32.96×** | 11.0 |
| `tiny-pool-dice` | 0.0030 | 0.0031 | 0.0036 ± 0.0017 | 47.8% / 0.0002 | +2.7% | inconclusive (`< 0.025 ms` floor) | 0.0092 | **2.95×** | 11.0 |
| `large-mean` | 0.0807 | 0.0813 | 0.0836 ± 0.0045 | 5.4% / 0.0035 | +0.7% | inconclusive (`< 0.025 ms` floor) | 0.1289 | **1.59×** | 118.7 |
| `large-variance` | 0.5175 | 0.5292 | 0.5374 ± 0.0197 | 3.7% / 0.0083 | +2.3% | inconclusive (`< 0.025 ms` floor) | 0.5055 | **0.96×** | 118.7 |
| `large-stddev` | 0.5178 | 0.5366 | 0.5402 ± 0.0093 | 1.7% / 0.0072 | +3.6% | inconclusive (`< 0.025 ms` floor) | 0.5070 | **0.94×** | 118.7 |
| `medium-student` | 0.2725 | 0.3072 | 0.3065 ± 0.0078 | 2.6% / 0.0129 | +12.7% | regression (`z=3.89`) | 0.6868 | **2.24×** | 22.6 |
| `medium-paired` | 0.2617 | 0.2945 | 0.2952 ± 0.0108 | 3.7% / 0.0057 | +12.5% | inconclusive | 0.5686 | **1.93×** | 23.2 |
| `medium-anova` | 0.4020 | 0.4667 | 0.4660 ± 0.0216 | 4.6% / 0.0318 | +16.1% | regression (`z=5.68`) | 0.8720 | **1.87×** | 23.1 |

---

## 3. Targeted `3.13.1` Patched-Path Microbenchmarks (Clean Idle Hardware, `3×7 + 2 warmups`)

Measured `2026-11-14T20:12:12Z` on idle Apple M4 Pro (Swift Release `-O -whole-module-optimization` vs Python `3.11.9` / `NumPy 2.3.5` / `Pandas 3.0.2` / `SciPy 1.17.1` / `Scikit-Learn 1.8.0` / `Statsmodels 0.14.6`). Earlier contended host measurements recorded `25.55 / 34.52 / 37.81 ms` writes and `12.041 ms` eager read (superseded by idle numbers below):

| Benchmark | Module | Swift Med (ms) | Swift Mean ± SD (ms) | Sw CV% / IQR | Sw Peak RSS (MB) | Python Med (ms) | Py CV% | Sw vs Py | Swift Checksum | Python Checksum |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| `CSV_Write_100k_NullTokens` | `SwiftDataFrame` | 15.714 | 15.697 ± 0.4650 | 3.0% / 0.4402 | 24.7 | 57.223 | 1.3% | **3.64×** | `bytes=2290455` | `bytes=2165456` |
| `CSV_Write_100k_PlainStrings` | `SwiftDataFrame` | 17.528 | 17.466 ± 0.4034 | 2.3% / 0.3563 | 27.9 | 60.423 | 0.6% | **3.45×** | `bytes=2830955` | `bytes=2830956` |
| `CSV_Write_100k_Numeric` | `SwiftDataFrame` | 9.878 | 9.925 ± 0.3516 | 3.5% / 0.5185 | 30.9 | 48.797 | 1.1% | **4.94×** | `bytes=1552950` | `bytes=1552951` |
| `CSV_Read_Eager_100k_NullTokens` | `SwiftDataFrame` | 3.512 | 3.536 ± 0.1769 | 5.0% / 0.2372 | 43.1 | 10.618 | 1.2% | **3.02×** | `rows=100000,cols=3` | `rows=100000,cols=3` |
| `CSV_Read_Stream_100k_NullTokens` | `SwiftDataFrame` | 3.326 | 3.352 ± 0.1706 | 5.1% / 0.0794 | 43.3 | 13.044 | 1.0% | **3.92×** | `rows=100000` | `rows=100000` |
| `TypedColumn_Unique_Double_100k` | `SwiftDataFrame` | 5.185 | 5.170 ± 0.1672 | 3.2% / 0.2630 | 43.3 | 0.2148 | 2.0% | **0.04×** | `count=248` | `count=247` |
| `DataFrame_Unique_100k_MixedKeys` | `SwiftDataFrame` | 50.534 | 50.258 ± 1.237 | 2.5% / 1.509 | 43.3 | 3.056 | 2.5% | **0.06×** | `rows=499` | `rows=499` |
| `TypedColumn_Variance_Standard_1M` | `SwiftDataFrame` | 8.134 | 8.118 ± 0.0881 | 1.1% / 0.0610 | 102.6 | 3.765 | 34.6% | **0.46×** | `v=0.995716,s=0.997856` | `v=0.995716,s=0.997856` |
| `TypedColumn_Variance_Shifted1e12_1M` | `SwiftDataFrame` | 8.095 | 8.085 ± 0.0863 | 1.1% / 0.1146 | 102.6 | 3.459 | 1.1% | **0.43×** | `v=9.957297e-05,s=9.978626e-03` | `v=9.957298e-05,s=9.978626e-03` |
| `Stats_ChiSquare_GOF_100` | `SwiftStats` | 0.0014 | 0.0014 ± 0.0001 | 4.7% / 0.0000 | 102.6 | 0.0903 | 7.1% | **65.67×** | `stat=156.274131,p=2.147307e-04` | `stat=156.274131,p=2.147307e-04` |
| `Stats_ShapiroWilk_100` | `SwiftStats` | 0.0033 | 0.0034 ± 0.0003 | 8.3% / 0.0001 | 102.7 | 0.0659 | 8.3% | **20.02×** | `w=0.973260,p=0.039465` | `w=0.973260,p=0.039465` |
| `Stats_ShapiroWilk_1000` | `SwiftStats` | 0.0303 | 0.0317 ± 0.0021 | 6.7% / 0.0034 | 102.7 | 0.0924 | 28.5% | **3.05×** | `w=0.998273,p=0.416374` | `w=0.998273,p=0.416374` |
| `Stats_ShapiroWilk_5000` | `SwiftStats` | 0.2078 | 0.2095 ± 0.0119 | 5.7% / 0.0129 | 102.8 | 0.2177 | 6.1% | **1.05×** | `w=0.999666,p=0.602935` | `w=0.999666,p=0.602935` |
| `Stats_KS_Lilliefors_200` | `SwiftStats` | 0.0072 | 0.0075 ± 0.0014 | 18.9% / 0.0001 | 102.8 | 0.0831 | 10.3% | **11.52×** | `d=0.044160,p=0.445784` | `d=0.044160,p=0.456809` |
| `Stats_KS_Lilliefors_2000` | `SwiftStats` | 0.0779 | 0.0783 ± 0.0045 | 5.7% / 0.0056 | 102.8 | 0.0865 | 3.8% | **1.11×** | `d=0.010317,p=0.872506` | `d=0.010317,p=0.886634` |
| `TargetEncoder_OOF_100k_100cats_cv5` | `SwiftPreprocessing` | 24.243 | 24.246 ± 0.3572 | 1.5% / 0.3146 | 103.1 | 9.263 | 1.0% | **0.38×** | `n=100000,sum=494635.5612` | `n=100000,sum=494630.2821` |
| `TargetEncoder_Matrix_OOF_10k` | `SwiftPreprocessing` | 2.740 | 2.745 ± 0.0160 | 0.6% / 0.0241 | 104.1 | 0.9408 | 1.4% | **0.34×** | `rows=10000,cols=1` | `rows=10000,cols=1` |
| `PolynomialFeatures_Deg4_1000x6` | `SwiftPreprocessing` | 0.6765 | 0.6788 ± 0.0078 | 1.2% / 0.0086 | 106.8 | 0.3165 | 1.7% | **0.47×** | `rows=1000,cols=210` | `rows=1000,cols=210` |
| `VarianceThreshold_1000x100_Shifted` | `SwiftPreprocessing` | 0.6570 | 0.6547 ± 0.0152 | 2.3% / 0.0214 | 108.6 | 0.3701 | 2.0% | **0.56×** | `kept=99` | `kept=99` |
| `SelectKBest_1000x100_k20` | `SwiftPreprocessing` | 0.5309 | 0.5325 ± 0.0074 | 1.4% / 0.0118 | 108.8 | 0.4263 | 2.2% | **0.80×** | `kept=20` | `kept=20` |
| `RFE_1000x50_k10_step2` | `SwiftPreprocessing` | 0.2592 | 0.2606 ± 0.0125 | 4.8% / 0.0199 | 109.5 | 16.173 | 1.8% | **62.40×** | `kept=10` | `kept=10` |

---

## 4. Historical Multi-Module Research Suite (`SwiftSciBenchmarks --research`, Retired in v3.13.2)

> [!WARNING]
> **Retired Legacy Archive Notice**: `SwiftSciBenchmarks --research` and `Benchmarks/Python/benchmarks.py --research` were **retired and removed in v3.13.2**. Section 4 below is preserved as an historical `3.13.1` archive only. Note specifically that `Holt-Winters fit (174.03×)` and `ARIMA(1,1,1) fit (172.57×)` compared non-comparable objectives (Hannan-Rissanen OLS vs Statsmodels Kalman MLE; SSE vs MLE), `Accuracy (310.90×)` measured aggregated console accuracy checks, and cumulative in-process `Peak RSS` cannot represent isolated operation heap. See [`PERFORMANCE/3.13.2.md`](../../../PERFORMANCE/3.13.2.md) for the published `3.13.2` validation.

| Benchmark | Module | Sw Mean ± CI95 (ms) | Sw Median (ms) | Sw Min..Max (ms) | Sw Mem (MB) | Py Median (ms) | Sw vs Py |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Mean (vDSP, 1M elements)` | SwiftStats | 0.1123 ± 0.0082 | 0.1213 | 0.0819..0.1371 | 24.7 | 0.1181 | **0.97×** |
| `StdDev (vDSP, 1M elements)` | SwiftStats | 0.6397 ± 0.0454 | 0.5763 | 0.5368..0.8119 | 24.8 | 0.4652 | **0.81×** |
| `Variance (vDSP, 1M elements)` | SwiftStats | 0.5399 ± 0.0017 | 0.5400 | 0.5311..0.5505 | 24.9 | 0.4548 | **0.84×** |
| `Pearson Correlation (500k)` | SwiftStats | 0.8391 ± 0.0094 | 0.8294 | 0.8227..0.9060 | 32.6 | 1.231 | **1.48×** |
| `Two-Sample T-Test (100k)` | SwiftStats | 0.3085 ± 0.0034 | 0.3065 | 0.2986..0.3223 | 35.0 | 0.4976 | **1.62×** |
| `Spearman Correlation (100k)` | SwiftStats | 12.018 ± 0.0948 | 11.972 | 11.838..12.923 | 42.1 | 15.003 | **1.25×** |
| `CSV Read (100k rows)` | SwiftDataFrame | 13.305 ± 0.0958 | 13.282 | 13.027..13.637 | 55.1 | 18.856 | **1.42×** |
| `CSV Stream Read (chunk=10k)` | SwiftDataFrame | 54.037 ± 0.2441 | 54.169 | 53.215..54.839 | 56.1 | 22.011 | **0.41×** |
| `CSV Stream + Filter` | SwiftDataFrame | 54.975 ± 0.2269 | 54.840 | 54.442..55.937 | 56.3 | 24.676 | **0.45×** |
| `CSV Stream + GroupBy` | SwiftDataFrame | 55.369 ± 0.2010 | 55.281 | 54.865..56.156 | 56.5 | 27.018 | **0.49×** |
| `Filter rows (100k)` | SwiftDataFrame | 0.6764 ± 0.0241 | 0.6632 | 0.5947..0.8059 | 61.6 | 0.6986 | **1.05×** |
| `GroupBy + Aggregation` | SwiftDataFrame | 1.685 ± 0.0038 | 1.684 | 1.669..1.714 | 61.6 | 1.582 | **0.94×** |
| `SortBy double column` | SwiftDataFrame | 2.719 ± 0.0377 | 2.690 | 2.615..2.907 | 65.4 | 7.159 | **2.66×** |
| `toFlatFeatureMatrix (100k rows)` | SwiftDataFrame | 0.1354 ± 0.0007 | 0.1355 | 0.1325..0.1385 | 65.4 | n/a | n/a |
| `Zero-Allocation df.rows iteration (100k)` | SwiftDataFrame | 3.980 ± 0.1153 | 3.924 | 3.621..4.959 | 65.4 | n/a | n/a |
| `DataFrame SIMD Hash Join (100k rows)` | SwiftDataFrame | 2.240 ± 0.1832 | 2.183 | 1.948..4.008 | 67.1 | 0.4249 | **0.19×** |
| `Parquet Write Snappy (100k rows)` | SwiftDataFrame | 9.184 ± 0.0772 | 9.184 | 9.145..9.223 | 70.1 | 10.529 | **1.15×** |
| `Parquet Read Snappy (100k rows)` | SwiftDataFrame | 24.099 ± 0.5458 | 24.099 | 23.821..24.378 | 74.6 | 1.457 | **0.06×** |
| `LinearRegression fit (10k×10, 100 epochs)` | SwiftML | 26.731 ± 4.356 | 23.584 | 22.814..52.614 | 92.0 | 3.607 | **0.15×** |
| `RandomForest fit (1k×4, 50 trees)` | SwiftML | 5.654 ± 0.1820 | 5.588 | 5.141..6.433 | 94.6 | 30.078 | **5.38×** |
| `GBDT Regressor fit (1k×4, 50 est.)` | SwiftML | 9.495 ± 0.1002 | 9.479 | 9.139..9.804 | 94.8 | 27.754 | **2.93×** |
| `KMeans fit (10k×4, 3 clusters)` | SwiftCluster | 19.089 ± 0.1035 | 19.094 | 18.738..19.349 | 95.4 | 7.232 | **0.38×** |
| `PCA SVD fitTransform (1k×100 → 10 comps)` | SwiftCluster | 0.7874 ± 0.0067 | 0.7833 | 0.7765..0.8313 | 96.5 | 0.7470 | **0.95×** |
| `IsolationForest fit (1k×10, 100 trees)` | SwiftCluster | 14.515 ± 0.1986 | 14.339 | 14.237..15.506 | 97.8 | 35.584 | **2.48×** |
| `LinearSVC fit (1k×4, 100 epochs, Metal GPU)` | SwiftML | 0.4001 ± 0.0115 | 0.3994 | 0.3782..0.4425 | 97.9 | 0.4185 | **1.05×** |
| `VectorStore Cosine Search (5k × 128d, top 10)` | SwiftCluster | 0.6415 ± 0.0077 | 0.6361 | 0.6275..0.7495 | 104.4 | 0.0270 | **0.04×** |
| `Holt-Winters fit (50k pts, period=12)` | SwiftForecast | 18.825 ± 0.1273 | 18.765 | 18.605..19.615 | 104.5 | 3265.6 | **Retired `--research` (`174.03×`)** |
| `ARIMA(1,1,1) fit (50k pts)` | SwiftForecast | 2.524 ± 0.1979 | 2.406 | 2.327..3.906 | 104.6 | 415.1 | **Retired `--research` (`172.57×`; OLS vs MLE)** |
| `ARIMA(1,1,1) forecast horizon=24 (50k pts)` | SwiftForecast | 2.550 ± 0.1078 | 2.530 | 2.389..3.277 | 104.9 | n/a | n/a |
| `Kalman Filter 1D (10k obs)` | SwiftForecast | 17.436 ± 0.1902 | 17.346 | 17.050..18.587 | 81.3 | n/a | n/a |
| `TS Decomposition additive (1k pts)` | SwiftForecast | 0.3158 ± 0.0108 | 0.3092 | 0.3089..0.4252 | 81.3 | 0.0751 | **0.24×** |
| `LLM Forward Pass (seqLen=64)` | SwiftLLM | 0.4139 ± 0.0179 | 0.4037 | 0.3798..0.4997 | 84.8 | n/a | n/a |
| `LLM Generate (10 tokens)` | SwiftLLM | 6.205 ± 0.1573 | 6.175 | 5.853..6.544 | 86.5 | n/a | n/a |
| `YOLOv8Detector detect (640×640 real GPU inference)` | SwiftVision | 28.897 ± 0.2658 | 28.859 | 28.043..29.651 | 135.3 | n/a | n/a |
| `YOLOPreprocessor letterbox (1920×1080 → 640×640)` | SwiftVision | 2.754 ± 0.0942 | 2.675 | 2.394..3.445 | 209.4 | n/a | n/a |
| `UNetSegmentationModel predict (128×128 image)` | SwiftVision | 2.029 ± 0.1455 | 1.936 | 1.724..2.540 | 214.8 | n/a | n/a |
| `KernelSHAP Explain (5 feats, 100 coalitions)` | SwiftExplain | 0.0347 ± 0.0069 | 0.0303 | 0.0222..0.0742 | 215.0 | 0.3704 | **12.23×** |
| `LIME Explain (5 feats, 300 samples)` | SwiftExplain | 0.0509 ± 0.0007 | 0.0505 | 0.0502..0.0559 | 215.0 | 0.3062 | **6.06×** |
| `Global Average Pooling & Dice Metric` | SwiftSci Extensions | 0.0033 ± 0.0007 | 0.0028 | 0.0027..0.0128 | 215.0 | 0.0063 | **2.25×** |
| `SQLite Direct DataFrame Ingestion` | SwiftSci Extensions | 0.0292 ± 0.0010 | 0.0281 | 0.0272..0.0408 | 215.1 | 0.0874 | **3.11×** |
| `RAG Context Summary Generation` | SwiftSci Extensions | 0.0002 ± 0.0000 | 0.0002 | 0.0002..0.0003 | 215.2 | 0.0006 | **2.79×** |
| `TreeSHAP Explanation (100 samples)` | SwiftSci Extensions | 0.3517 ± 0.0049 | 0.3444 | 0.3427..0.3882 | 215.3 | 0.0622 | **0.18×** |
| `OneVsRestClassifier (5 classes, 100 samples)` | SwiftSci Extensions | 0.8954 ± 0.0518 | 0.8832 | 0.7007..1.297 | 215.3 | 4.203 | **4.76×** |
| `TF-IDF Vectorizer (50 documents)` | SwiftSci Extensions | 0.3735 ± 0.0053 | 0.3693 | 0.3663..0.4316 | 215.4 | 0.3563 | **0.96×** |
| `Forecast Errors Suite (RMSE, MAE, MAPE, R² 100k)` | SwiftSci Extensions | 0.9391 ± 0.0087 | 0.9359 | 0.9043..0.9935 | 215.4 | 0.5911 | **0.63×** |
| `Classification ROC-AUC (50k predictions)` | SwiftSci Extensions | 2.878 ± 0.0437 | 2.879 | 2.754..3.042 | 215.5 | 5.669 | **1.97×** |
| `OneHotEncoder fitTransform (50k rows)` | SwiftSci Extensions | 5.542 ± 0.0711 | 5.505 | 5.314..5.807 | 224.7 | 20.694 | **3.76×** |
| `VADER Sentiment Analysis (1k sentences)` | SwiftSci Extensions | 9.892 ± 0.0471 | 9.898 | 9.603..10.118 | 224.9 | n/a | n/a |
| `NaiveBayesClassifier fit (1k×100, 3 classes)` | SwiftSci Extensions | 0.0268 ± 0.0007 | 0.0259 | 0.0252..0.0318 | 224.9 | 0.5077 | **19.64×** |
| `Holt-Winters Forecast (RMSE, MAE, R²)` | SwiftSci Accuracy | 0.4780 ± 0.1457 | 0.3625 | 0.3539..1.022 | 225.0 | 50.417 | **Retired `--research` (`139.07×`)** |
| `ARIMA(1,1,1) Forecast (RMSE, MAE, R²)` | SwiftSci Accuracy | 0.0359 ± 0.0006 | 0.0355 | 0.0353..0.0381 | 225.0 | 11.024 | **Retired `--research` (`310.90×`; OLS vs MLE)** |
| `OLS Linear Regression (LAPACK)` | SwiftSci Accuracy | 0.0762 ± 0.0023 | 0.0740 | 0.0727..0.0810 | 225.0 | n/a | n/a |
| `GBDT Regressor Quality (RMSE, R²)` | SwiftSci Accuracy | 4.853 ± 0.1263 | 4.828 | 4.642..5.152 | 225.3 | 10.146 | **2.10×** |
| `HistGBDT Regressor (RMSE, R²)` | SwiftSci Accuracy | 1.433 ± 0.0646 | 1.426 | 1.307..1.590 | 225.4 | n/a | n/a |
| `DecisionTree Regressor (RMSE, R²)` | SwiftSci Accuracy | 0.2534 ± 0.0092 | 0.2465 | 0.2425..0.2817 | 225.4 | n/a | n/a |
| `LogisticRegression (Accuracy, F1)` | SwiftSci Accuracy | 1.765 ± 0.0266 | 1.756 | 1.724..1.860 | 225.5 | n/a | n/a |
| `LinearSVC (Accuracy, F1)` | SwiftSci Accuracy | 0.8600 ± 0.2397 | 0.6902 | 0.6148..1.713 | 225.5 | n/a | n/a |
| `RandomForest Classifier (Accuracy, F1)` | SwiftSci Accuracy | 2.544 ± 0.1667 | 2.428 | 2.275..2.921 | 226.5 | 14.414 | **5.94×** |
| `HistGBDT Classifier (Accuracy, F1)` | SwiftSci Accuracy | 1.236 ± 0.0570 | 1.204 | 1.152..1.413 | 226.5 | n/a | n/a |
| `DecisionTree Classifier (Accuracy, F1)` | SwiftSci Accuracy | 0.4803 ± 0.0260 | 0.4680 | 0.4463..0.5617 | 226.6 | n/a | n/a |
| `NaiveBayes (Accuracy, Macro-F1)` | SwiftSci Accuracy | 0.0453 ± 0.0002 | 0.0452 | 0.0449..0.0457 | 227.1 | 0.6201 | **13.72×** |
| `PCA SVD Spectrum (EVR Parity)` | SwiftSci Accuracy | 0.0262 ± 0.0010 | 0.0254 | 0.0253..0.0296 | 227.1 | n/a | n/a |
| `K-Means Convergence (WCSS Inertia)` | SwiftSci Accuracy | 0.1027 ± 0.0059 | 0.1055 | 0.0906..0.1181 | 227.1 | n/a | n/a |
| `StandardScaler Parity (vDSP)` | SwiftSci Accuracy | 0.0538 ± 0.0025 | 0.0536 | 0.0495..0.0627 | 227.1 | n/a | n/a |
| `MinMaxScaler Parity (vDSP)` | SwiftSci Accuracy | 0.0678 ± 0.0053 | 0.0643 | 0.0610..0.0856 | 227.1 | n/a | n/a |
| `Hypothesis Tests & Correlation Suite` | SwiftSci Accuracy | 0.0800 ± 0.0013 | 0.0793 | 0.0783..0.0846 | 227.1 | n/a | n/a |
| `VADER Sentiment Polarity Analysis` | SwiftSci Accuracy | 0.0246 ± 0.0002 | 0.0245 | 0.0243..0.0254 | 227.1 | n/a | n/a |
