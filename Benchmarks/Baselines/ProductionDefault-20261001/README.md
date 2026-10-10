# Initial production-default performance baseline

Measured commit [`980884dd4c755e19ff0def721ab8b2db3c6343bf`](https://github.com/crabel99/SwiftSci/commit/980884dd4c755e19ff0def721ab8b2db3c6343bf). Source fingerprint `43eb2dcd02252e033f3e6472fb55cb7a1891fb03b19491e7c270a552abc5e6cc`.

All six performance profiles passed and audited: 2,060 fresh worker executions and 10,300 measured samples. Each case used five processes per engine, two warmups and five measured samples per process. Engine order rotated each round. Source identity remained unchanged throughout the suite.

## Representative timings

Milliseconds, median of five process medians. These rows have resolved timings. Each row uses the same declared workload contract across its engines.

| Operation | SwiftSci | pandas / NumPy | Polars | DuckDB |
|---|---:|---:|---:|---:|
| CSV read, 1M rows | 14.792 | 94.504 | 5.148 | 54.546 |
| Filter, 1M rows | 2.191 | 4.769 | 1.083 | 6.011 |
| Sort, 1M rows | 30.142 | 32.882 | 10.665 | 18.117 |
| Group sum, 1M rows | 3.690 | 3.802 | 2.905 | 0.517 |
| Matrix export, 1M rows | 1.592 | 2.773 | 2.451 | 5.741 |
| Target export, 1M rows | 0.788 | 0.222 | 0.620 | 3.856 |
| Standard scaling, 1M rows | 28.896 | 4.564 | 3.537 | 9.830 |
| Min-max scaling, 1M rows | 8.546 | 3.713 | 3.532 | 10.226 |
| Numeric join, 100k rows | 31.887 | 0.887 | 2.068 | 2.189 |
| Mixed-type join, 100k rows | 37.066 | 0.909 | 2.134 | 3.002 |
| Parquet read, 100k rows | 24.056 | 1.980 | 1.995 | 3.275 |

## Interpretation

SwiftSci is competitive with pandas on several common operations and leads the measured matrix export case. Native-threaded Polars is faster on CSV reading, filtering and sorting; DuckDB leads the shown group sum. Joins remain about 36–41 times slower than pandas, and Parquet reading is about 12 times slower than pandas or Polars. Standard scaling is about eight times slower than Polars. These are the strongest candidates for profiling.

These measurements do not establish a universal engine ranking. They use warm input files and the existing operation and workflow contracts. They exclude process startup from operation timings. This suite does not measure pretrained LLM throughput or memory-pressure workflows.

## Correctness qualification

Raw CPU conformance status: `failed`. Coverage complete: `True`.

- swiftsci: 202 passed, 0 failed.
- pandas: 198 passed, 4 failed.

The regression policy passed with no new blockers. The four retained Python failures are Wampler4 and Wampler5 under binary64 and original-decimal contracts. See the unchanged [CPU acceptance inventory](cpu-acceptance.json) and [regression policy](cpu-regression-policy.json). Raw conformance remains failed; no tolerance was relaxed.

## Conditions and limits

Workers used native threading defaults. Polars and DuckDB reported 16-thread pools throughout the full performance run. Pool size does not measure active thread use. AC power and power settings remained unchanged within every performance profile.

This Mac had active desktop and background processes. Unrelated process listings remain in the private local archive and are omitted from this export. Small differences should be checked on an isolated run before driving design changes. Some engine/case timings varied between processes; all medians, ranges, standard deviations and raw samples are retained. No outliers were removed.

There are 32 timing-unresolved summaries among 412 engine/case summaries. They remain correctness evidence and do not support speedup claims. Whole-process peak RSS includes setup and validation; it is not an operation's allocation cost.

See [all 412 summaries](measurements.json) and the [export manifest](manifest.json). Each profile directory contains every measured sample in `events.jsonl` and its resolved experiment in `metadata.json`.

## Evidence format and verification

This is a sanitized export of the completed run, not a newly executed benchmark. The exporter changes only the user-home prefix and hostname in metadata. Timing samples, output hashes, validation errors, memory peaks, contracts and source fingerprints are unchanged. The original archive remains unmodified.

Each `certificate.original.json` binds its original unredacted `run.json` by SHA-256. Those original run hashes are retained in the manifest. The sanitized files are not the original certificate's byte-for-byte input. Export-file hashes verify the committed package; they are unsigned integrity records, not independent accreditation.

From the repository root, run the export verifier:

```sh
python3 Benchmarks/Baselines/ProductionDefault-20261001/verify.py
```

The verifier checks export hashes, reconstructs the six in-memory run records, audits their contracts and samples, recomputes every summary, checks power and pool consistency, and validates the retained CPU-policy records. Generated Parquet files and complete CPU worker runs are not included. Their original audit results and hashes are retained; verifying this export does not repeat file-level Parquet validation or independently rerun numerical calculations.

The measured machine was an Apple M4 Max with 128 GiB memory, macOS 27.0 build 26A428, Swift 6.4 and Xcode 27.0. Workers were native arm64 Release builds without coverage instrumentation. Metal used safe math with precise FP32 functions. Full compiler, binary, library and reference-engine identities are in each profile's metadata.

## Reproduce and use this baseline

Use the measured revision linked above, pinned comparison requirements, and the [standardized runner](../../README.md). Build an uninstrumented worker, then run:

```sh
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/comparison_suite.py \
  --mode production-default \
  --swift-worker "$worker" \
  --python Benchmarks/.venv-standardized/bin/python \
  --output Benchmarks/Runs/production-default-new-run
```

Use a new output directory. The comparison driver prepares its fixtures and runs CPU qualification after the six performance profiles. CPU qualification retains its bounded legacy execution policy; its diagnostic timings are not part of the production-default performance comparison.

Use this record to choose profiling targets and retain it when evaluating subsequent changes. Rerun baseline and candidate with matching experiment contracts, dependencies, hardware and power settings before attributing a difference to a code change. Small timing differences from this active desktop run do not establish an improvement or a CI performance threshold.
