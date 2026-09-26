# Dataframe optimization evidence

Measured implementation: `4c5bb953547ba5d84c7fd86bf868b1680c5f1811` on `codex/dataframe-pipeline-optimization`. Evidence recorded September 25 and 26, 2026 on an Apple M4 Max with 128 GiB unified memory, macOS 27.0 and Swift 6.4. This directory was added after testing; it does not change production code.

The changes reduce repeated type dispatch, conversion, null scanning and intermediate allocation in CSV reads, filters, sorts and grouped reductions. Public optional-array column storage remains in place. Compact storage and its AI integration research are separate work.

## Implementation scope

- Dispatch built-in numeric filters outside the row loop and preserve exact mixed numeric comparisons. Gather by concrete type, carry known null counts, and schedule parallel gathering by estimated output bytes.
- Cache typed sort keys and preserve stable ties and null placement. Large Double inputs can use radix sorting, with comparison fallbacks for ordered, sparse and NaN-containing inputs.
- Represent groups with flat first-seen group IDs. Use bounded lookup/refinement for suitable integer domains and typed component identity for general composite keys. Avoid per-row string key construction.
- Use compensated floating sums and means. Add opt-in `sumChecked()` for exact integer totals with a checked final Int64 result.
- Flatten CSV field metadata, keep mapped-byte access within its lifetime, scan eligible unquoted files in parallel, allocate final indexes once, and fuse null counting with column construction. Decimal conversion rounds complete significands once.
- Retain the CPU/Metal indexing comparison as an opt-in benchmark. Production CSV does not dispatch to the GPU.

Behavior changes are documented in [numeric filtering](../../../Sources/SwiftDataFrame/SwiftDataFrame.docc/NumericFiltering.md), [group equality](../../../Sources/SwiftDataFrame/SwiftDataFrame.docc/GroupKeySemantics.md), and [checked sums](../../../Sources/SwiftDataFrame/SwiftDataFrame.docc/CheckedIntegerSums.md).

## Matched improvement measurements

These are separate before/after experiments on identical fixtures within each experiment. They establish individual stages, not a cumulative multiplier. Baselines and candidate hashes are retained in the linked evidence. All rows below use one million input rows.

| Stage | Before ms | After ms | Improvement | Evidence |
|---|---:|---:|---:|---|
| Initial typed/cached sorting | 213.743 | 68.419 | 3.12x | [Raw samples](early-stages/sort-cached-keys.json) |
| Typed integer group keys | 249.548 | 22.548 | 11.07x | [Raw samples](early-stages/group-typed-keys.json) |
| Flat group IDs | 24.090 | 18.936 | 1.27x | [Raw samples](early-stages/dense-ids.json) |
| Bounded integer lookup | 18.490 | 3.962 | 4.67x | [Raw samples](early-stages/bounded-lookup.json) |
| Integrated CSV layout/null matching | 111.861 | 39.339 | 2.84x | [Raw samples](integrated-before-after/raw.json) |
| Adaptive Double sorting | 65.674 | 36.288 | 1.81x | [Raw samples](integrated-before-after/raw.json) |
| Bounded two-key grouping | 49.536 | 11.375 | 4.35x | [Raw samples](integrated-before-after/raw.json) |
| Integrated filter → sort → group | 36.730 | 20.077 | 1.83x | [Raw samples](integrated-before-after/raw.json) |
| Subsequent CSV scan/allocation changes | 38.947 | 13.565 | 2.87x | [Raw samples](csv-before-after/results.json) |
| Final Float64 filtering changes | 3.237 | 1.944 | 1.66x | [Summary](filter-before-after/summary.json) |
| Final native Int filtering changes | 3.737 | 1.951 | 1.92x | [Summary](filter-before-after/summary.json) |
| Final Int32 filtering changes | 3.723 | 1.947 | 1.91x | [Summary](filter-before-after/summary.json) |
| Final Float32 filtering changes | 3.768 | 1.985 | 1.90x | [Summary](filter-before-after/summary.json) |

The initial typed-column patch also contains the historical roughly 8x filter improvement measured during profiling. The PR's quantitative justification uses the paired evidence above and the final comparison below rather than multiplying historical measurements from different runs.

Early sort/group experiments used three alternating process pairs, two warmups and fifteen timed repetitions per process. The later integrated, CSV and final-filter experiments used three process pairs, two warmups and five timed repetitions per process. Medians pool the samples. Complete output checks accompany the comparisons; CSV comparison allows relative error of 1e-15 for the intentional decimal-rounding correction.

The integrated experiment compared preserved baseline `2694d303f8` with the adaptive dense-input candidate before its final sparse-density guard. Its fixtures have about 99% present values; the guard only changes very sparse inputs. Candidate metadata records the pre-commit source state and executable hashes. The clean final implementation was subsequently compared independently in the production run below.

The later CSV stage reduced whole-process peak RSS from 310.1 to 186.4 MiB, about 40%. The preceding CSV layout stage reduced it from 584.4 to 308.5 MiB. These measurements include fixtures, retained results and allocator caches. They are not isolated allocation counts and must not be added. Final filtering improved speed with essentially unchanged peak RSS.

The final filter baseline is `4dc6b533dd`; candidate is `4c5bb95354`. Its [metadata](filter-before-after/metadata.json) and [per-process measurements](filter-before-after/measurements.jsonl) identify the executable hashes and exact output comparisons.

## Final comparison with pandas and Kiraa

Native arm64 Release implementations, two warmups and five timed repetitions in each of three fresh processes per library. Library order rotates, and all workloads run serially. Table values are median milliseconds at one million rows.

| Operation | SwiftSci | pandas / Python | Kiraa |
|---|---:|---:|---:|
| CSV read | 13.560 | 67.064 | 9.171 |
| Float64 filter | 1.940 | 2.121 | 2.830 |
| Native Int / Int64 filter | 1.971 | 1.846 | INVALID |
| Int32 filter | 1.949 | 1.782 | n/a |
| Float32 filter | 1.983 | 2.253 | n/a |
| Stable descending sort | 31.367 | 29.229 | 58.971 |
| Single-key group sum | 4.663 | 4.896 | 9.361 |
| Two-key group sum | 11.336 | 10.599 | 36.498 |
| Filter → sort → group sum | 17.012 | 20.118 | 41.025 |
| Mean + sample variance | 0.587 | 0.608 | 0.590 |
| Correlation | 1.713 | 2.421 | n/a |
| Exponential smoothing + forecast | 12.605 | 388.562 | n/a |

[Full comparison](production/REPORT.md) includes 100,000-row results, timing spread, peak RSS, exact workload definitions and API differences. [Raw measurements](production/measurements.json) retain all samples and parity results. [Metadata](production/metadata.json) records source, toolchain, binary and fixture hashes.

All 24 SwiftSci workload/size combinations passed output comparisons against pandas/Python. The two Kiraa integer-filter cases failed parity and are excluded from rankings. Unsupported Kiraa operations are marked n/a. Statistics and correlation compare against NumPy; forecasting compares against statsmodels and has different auxiliary diagnostics. These unmodified numerical operations provide context, not performance gains attributable to this PR.

Kiraa remains faster on the recorded CSV fixture. pandas remains faster on sorting, two-key grouping and the million-row integer filters. At 100,000 rows, two-key grouping is 4.107 ms for SwiftSci, 1.658 ms for pandas and 3.977 ms for Kiraa. No universal superiority claim is supported.

## Correctness and broader validation

- Complete Debug suite: 891 passed, zero failed or skipped. [Summary](tests/debug.json).
- Complete Release suite: 891 passed, zero failed or skipped. [Summary](tests/release.json).
- Each full run includes MLX-dependent targets and 978 device-level executions after parameter expansion.
- Filter tuning covers 96 layouts; a separate holdout covers 24 layouts with different sizes, widths, seeds and String/Bool payloads. All selected rows, schema, values and null counts were checked independently. [Tuning](filter-breadth/tuning-revised.json), [holdout](filter-breadth/holdout.json).
- All 96 operator cases passed independent selected-index checks across numeric types, comparison/null operators, fractional thresholds, NaNs and infinities. [Operator results](filter-breadth/operators-revised.json), [cached-null results](filter-breadth/operators-cached.json).
- Parser AddressSanitizer and ThreadSanitizer runs each passed 764 cases over 87,060,376 bytes at the CSV-stage source. These instrumented checks covered pointer bounds and races, not leak detection. [Address result](tests/address/results.txt), [thread result](tests/thread/results.txt). The later filter commits did not change parser source.
- Decimal regression coverage includes binary64 bit patterns, long decimals, scientific notation, signed zero, overflow/underflow and integer typing beyond 2^53.

The initial holdout recorded two slowdowns of 10.6% and 6.1%. A five-round follow-up without code changes measured those cases as 0.7% slower and 2.7% faster. The original results remain included; the holdout was not used to retune. The final geometric-mean improvement across the 20 holdouts above 0.05 ms was 1.129x. [Confirmation results](filter-breadth/holdout-confirmation.json).

Full suites ran with the native Xcode test runner and a build cache outside Documents to avoid signing metadata and MLX resource-location problems. Release testing enabled testability without changing optimization. This is local validation, not a claim that remote CI ran on the pushed branch. Reproduction commands are in the [comparison instructions](comparison/README.md).

## Limits and scope

The machine was shared with other applications. Larger repeated improvements are stronger evidence than small ranking differences. Measurements use warm inputs and exclude imports, fixture construction, result extraction and serialization. RSS is whole-process peak memory. Library-internal parallelism is preserved, with BLAS thread limits set to one where honored.

CSV throughput uses three numeric columns and no quoted fields. Quoted/ragged/custom-delimiter inputs have correctness tests, but equivalent speedups are not established for those distributions. Kiraa infers some numeric CSV fields as Double where SwiftSci preserves integers. Exact physical storage is therefore not identical.

The cross-library checker normalizes fixture numbers through Float64 with rtol 1e-10 and atol 1e-8. Fixture integers fit exactly; dedicated regression tests cover larger integer boundaries. The correlation benchmark uses non-null arrays and does not validate every dataframe conversion path. The separate architecture review identified existing conversion defects outside this patch.

Existing CSV integer-parser overflow, sampled type inference and trailing-delimiter behavior remain outside this change. Existing NaN sorting fallback remains in place. The current optional-array storage API is unchanged. No production GPU CSV acceleration, compact storage replacement or AI inference speedup is claimed.

The standalone GPU experiment found usable index construction slower than parallel CPU scanning, so it was not integrated. Its source and methodology are in [CSVAcceleration](../../CSVAcceleration/README.md).

## Evidence provenance

Machine-specific path prefixes and test device identifiers were normalized before publication. Process-name snapshots, binaries, large generated fixtures, personal editor settings and unrelated research are omitted. [Artifact hashes](artifact-hashes.json) record original and published hashes. Numerical samples and parity outcomes are preserved.

The comparison workers are byte-for-byte copies of the measured Swift and Python sources. The relocated runner changes path configuration and omits process-name capture; it does not change timing or parity logic. Generated CSVs reproduce both recorded SHA-256 hashes. The later portable-runner smoke check is separate from the recorded performance results.
