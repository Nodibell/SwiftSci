# Production comparison: SwiftSci, pandas and Kiraa

Run 20260926T104626Z on Apple M4 Max, 128 GiB unified memory. SwiftSci `4c5bb953547b`; Kiraa `fc5d957401f7`. pandas 3.0.6, NumPy 2.5.3.

These are the real library APIs in native arm64 Release builds, using the committed SwiftSci optimization branch. The compact-storage prototype is excluded. The source trees had no tracked modifications. No production source changed during this comparison.

## Runtime

Values are medians in milliseconds. Each workload has three separately launched processes per library, two warmups per process and five timed repetitions per process: 15 samples in total. Library order rotates between batches. Compilation finishes before timing. Workloads run sequentially.

### 100,000 rows

| Operation | SwiftSci ms | pandas / Python ms | Kiraa ms |
|---|---:|---:|---:|
| Read CSV | 5.465 | 7.141 | 2.655 |
| Filter Float64 | 0.224 | 0.290 | 0.294 |
| Filter native Int / Int64 | 0.212 | 0.267 | INVALID |
| Filter Int32 | 0.214 | 0.264 | n/a |
| Filter Float32 | 0.215 | 0.297 | n/a |
| Stable descending sort | 2.782 | 2.297 | 4.589 |
| Group sum, 100 keys | 0.484 | 0.690 | 0.896 |
| Group sum, 9,700 key pairs | 4.107 | 1.658 | 3.977 |
| Filter → sort → group sum | 1.662 | 1.904 | 3.443 |
| Mean + sample variance | 0.061 | 0.069 | 0.060 |
| Pearson correlation | 0.175 | 0.272 | n/a |
| Exponential smoothing + 24 forecasts | 1.273 | 39.645 | n/a |

### 1,000,000 rows

| Operation | SwiftSci ms | pandas / Python ms | Kiraa ms |
|---|---:|---:|---:|
| Read CSV | 13.560 | 67.064 | 9.171 |
| Filter Float64 | 1.940 | 2.121 | 2.830 |
| Filter native Int / Int64 | 1.971 | 1.846 | INVALID |
| Filter Int32 | 1.949 | 1.782 | n/a |
| Filter Float32 | 1.983 | 2.253 | n/a |
| Stable descending sort | 31.367 | 29.229 | 58.971 |
| Group sum, 100 keys | 4.663 | 4.896 | 9.361 |
| Group sum, 9,700 key pairs | 11.336 | 10.599 | 36.498 |
| Filter → sort → group sum | 17.012 | 20.118 | 41.025 |
| Mean + sample variance | 0.587 | 0.608 | 0.590 |
| Pearson correlation | 1.713 | 2.421 | n/a |
| Exponential smoothing + 24 forecasts | 12.605 | 388.562 | n/a |

## Correctness and API differences

Every output column and value is compared with pandas/Python after each batch. Missing values must occupy the same positions. Numeric tolerance is rtol=1e-10 and atol=1e-8. Integer values and group sums in these fixtures fit exactly within Float64, which the checker uses for normalization. Filtering and sorting retain row-order checks; grouped rows are aligned by keys outside timing. This benchmark supplements, rather than replaces, the exact integer boundary regression tests.

Kiraa integer filtering failed parity: the tested ColumnPredicate/Series path returns an all-false mask for an Int64 column. Its Series comparison implementation only accepts double storage. Invalid timings are excluded from rankings.

Kiraa has no matching scalar Int32 or Float32 column storage in this checkout. No equivalent correlation or forecasting API was identified, so those cases compare SwiftSci with Python only.

Kiraa composite grouping preserves both original key columns and uses ordinal index labels. The adapter reads those actual columns for parity checks. SwiftSci emits group keys as strings, which are converted to numeric keys outside timing. Kiraa single-key grouping uses string index labels.

CSV inference also differs: Kiraa represents the numeric ID and group fields as Float64, while pandas retains integer columns. Value parity is checked, but identical physical types are not claimed.

## Workloads

- The baseline dataframe has integer ID and group columns, Float64 values, 100 groups, and a missing value every 101 rows. Values follow ((id × 7919) mod 100003) / 100. Both sizes use the same deterministic construction.
- Float filtering selects values greater than 500. Integer filtering uses the numerator and a threshold of 50000, with the same null positions. Swift native Int and Kiraa Int64 use 64-bit integers on this Mac; pandas uses nullable Int64. The Int32 case uses nullable Int32 in pandas. The Float32 case rounds all values to Float32 before filtering.
- Sorting is descending, stable, with nulls last. Grouping includes key factorization and summing both numeric value columns. Two-key grouping adds id mod 97, yielding 9,700 populated key pairs. The pipeline measures the combined public filter, sort and group calls.
- CSV reads use identical input files and a warm filesystem cache. File hashes are recorded.
- Statistics use the original non-null Double array. Python uses NumPy for mean, sample variance and correlation; forecasting uses statsmodels. The smoothing parameter is fixed at 0.3, initialization uses the first observation, optimization is disabled, and 24 point forecasts are compared. The forecasting APIs also calculate different auxiliary diagnostics.
- Timings exclude imports, input construction, result extraction, parity checks and JSON serialization. Results remain alive until timing ends. This is library-level execution, not a CLI or daemon benchmark.

## Memory

Median whole-process peak RSS across three launches, in MiB, sampled before JSON result extraction. It includes imports, setup arrays, input frames, retained results and allocator caches across repetitions. It is not incremental operation memory, nor a minimal-service memory estimate. The workers retain some fixture arrays that an application could release. statsmodels is imported only for forecasting.

| Rows | Operation | SwiftSci MiB | pandas / Python MiB | Kiraa MiB |
|---:|---|---:|---:|---:|
| 100,000 | Read CSV | 33.5 | 120.0 | 21.4 |
| 100,000 | Filter Float64 | 17.2 | 79.8 | 11.7 |
| 100,000 | Filter native Int / Int64 | 17.1 | 80.5 | INVALID |
| 100,000 | Filter Int32 | 15.7 | 80.6 | n/a |
| 100,000 | Filter Float32 | 14.6 | 79.1 | n/a |
| 100,000 | Stable descending sort | 20.6 | 82.1 | 12.5 |
| 100,000 | Group sum, 100 keys | 13.6 | 80.3 | 17.6 |
| 100,000 | Group sum, 9,700 key pairs | 16.4 | 84.1 | 19.0 |
| 100,000 | Filter → sort → group sum | 19.3 | 81.8 | 19.7 |
| 100,000 | Mean + sample variance | 13.5 | 78.4 | 10.9 |
| 100,000 | Pearson correlation | 14.2 | 79.2 | n/a |
| 100,000 | Exponential smoothing + 24 forecasts | 16.7 | 168.4 | n/a |
| 1,000,000 | Read CSV | 186.4 | 337.7 | 253.3 |
| 1,000,000 | Filter Float64 | 98.3 | 147.1 | 65.0 |
| 1,000,000 | Filter native Int / Int64 | 113.6 | 157.8 | INVALID |
| 1,000,000 | Filter Int32 | 98.3 | 156.0 | n/a |
| 1,000,000 | Filter Float32 | 98.3 | 144.3 | n/a |
| 1,000,000 | Stable descending sort | 136.8 | 201.9 | 80.5 |
| 1,000,000 | Group sum, 100 keys | 75.5 | 138.3 | 67.8 |
| 1,000,000 | Group sum, 9,700 key pairs | 99.6 | 198.8 | 77.6 |
| 1,000,000 | Filter → sort → group sum | 125.2 | 174.4 | 88.8 |
| 1,000,000 | Mean + sample variance | 75.3 | 134.3 | 53.4 |
| 1,000,000 | Pearson correlation | 83.0 | 149.9 | n/a |
| 1,000,000 | Exponential smoothing + 24 forecasts | 98.4 | 328.8 | n/a |

## Timing spread

Interquartile ranges over the 15 samples, in milliseconds. These describe variation within this run, not confidence intervals across machines.

| Rows | Operation | SwiftSci Q1 to Q3 | pandas / Python Q1 to Q3 | Kiraa Q1 to Q3 |
|---:|---|---:|---:|---:|
| 100,000 | Read CSV | 5.406 to 5.505 | 6.730 to 7.283 | 2.612 to 2.736 |
| 100,000 | Filter Float64 | 0.214 to 0.233 | 0.281 to 0.298 | 0.292 to 0.294 |
| 100,000 | Filter native Int / Int64 | 0.208 to 0.216 | 0.263 to 0.275 | INVALID |
| 100,000 | Filter Int32 | 0.209 to 0.223 | 0.257 to 0.270 | n/a |
| 100,000 | Filter Float32 | 0.211 to 0.223 | 0.292 to 0.308 | n/a |
| 100,000 | Stable descending sort | 2.773 to 2.816 | 2.283 to 2.339 | 4.578 to 4.649 |
| 100,000 | Group sum, 100 keys | 0.477 to 0.486 | 0.684 to 0.706 | 0.893 to 0.908 |
| 100,000 | Group sum, 9,700 key pairs | 4.004 to 4.134 | 1.597 to 1.733 | 3.930 to 4.061 |
| 100,000 | Filter → sort → group sum | 1.657 to 1.693 | 1.880 to 1.938 | 3.414 to 3.472 |
| 100,000 | Mean + sample variance | 0.059 to 0.063 | 0.067 to 0.072 | 0.058 to 0.062 |
| 100,000 | Pearson correlation | 0.174 to 0.175 | 0.265 to 0.276 | n/a |
| 100,000 | Exponential smoothing + 24 forecasts | 1.225 to 1.324 | 39.542 to 40.018 | n/a |
| 1,000,000 | Read CSV | 13.283 to 13.871 | 66.765 to 67.908 | 9.084 to 9.319 |
| 1,000,000 | Filter Float64 | 1.937 to 1.943 | 2.101 to 2.162 | 2.819 to 2.858 |
| 1,000,000 | Filter native Int / Int64 | 1.956 to 1.994 | 1.825 to 1.939 | INVALID |
| 1,000,000 | Filter Int32 | 1.943 to 1.951 | 1.778 to 1.792 | n/a |
| 1,000,000 | Filter Float32 | 1.975 to 1.995 | 2.239 to 2.353 | n/a |
| 1,000,000 | Stable descending sort | 30.937 to 31.681 | 29.024 to 29.730 | 58.747 to 59.616 |
| 1,000,000 | Group sum, 100 keys | 4.644 to 4.675 | 4.747 to 4.947 | 9.343 to 9.821 |
| 1,000,000 | Group sum, 9,700 key pairs | 11.298 to 11.479 | 10.459 to 10.642 | 36.105 to 36.960 |
| 1,000,000 | Filter → sort → group sum | 16.884 to 17.108 | 20.020 to 20.264 | 40.422 to 42.275 |
| 1,000,000 | Mean + sample variance | 0.586 to 0.590 | 0.608 to 0.610 | 0.586 to 0.609 |
| 1,000,000 | Pearson correlation | 1.711 to 1.716 | 2.374 to 2.471 | n/a |
| 1,000,000 | Exponential smoothing + 24 forecasts | 12.144 to 13.031 | 386.741 to 389.300 | n/a |

## Interpretation limits

Other applications were actively using several CPU cores during the run. Process-name CPU snapshots were retained privately and are omitted from this publication. These results describe a shared M4 Max, not an isolated thermal or power-controlled machine. Close other workloads before treating small differences as stable rankings.

BLAS thread limits are set to one where honored. They do not restrict Kiraa’s parallel CSV parser. All SWIFTPANDAS overrides are cleared. Both row counts are below Kiraa’s recorded 10-million-row GPU-group threshold. This compares each library’s default CPU paths, including its internal parallelism. It does not measure MLX inference or GPU model performance.

This run improves two details from the earlier overlap harness: pandas filtering uses its ordinary selection result without a redundant .copy(), and statsmodels loads only for forecasting. Consequently, compare libraries within this run; older Python filtering, startup and memory figures are not directly comparable. Swift packages remain in their own declared language modes; the Kiraa client uses Swift 5 mode to match its original benchmark.

## Reproduce

Use the [comparison instructions](../comparison/README.md). The timing and parity workers are the recorded sources. The portable runner changes local paths and omits process-name snapshots; it retains workload, timing, rotation and parity logic. These recorded results were not regenerated using that relocated runner.

`metadata.json` preserves source revisions, original binary/source hashes, input hashes and toolchain. `measurements.json` contains every sample and per-batch parity results. Machine-specific path prefixes were normalized. See [artifact hashes](../artifact-hashes.json) for original and published file hashes.
