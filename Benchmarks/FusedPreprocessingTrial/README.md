# Fused preprocessing trial

This experiment asks whether fitted imputation, standard scaling, and row-major packing benefit from sharing a bounded pass over compact columns. It is confined to the benchmark worker and does not add a public preprocessing API.

The trial starts at Core ML PR #65 head `62c3d7e61146db1b5fc36b39c7d4e424f0e38572`. Its native prepared-data baseline imports the preprocessing implementation and related tests from CPU/GPU PR #63 head `9d7bc22363aef7e94ef910b36d109113e6c5bb21`. That avoids comparing fusion against the older row-array adapter. The import is a separate commit so the experimental changes remain identifiable.

## Compared paths

- Staged native imputation, consuming standard scaling, then direct packing into Float32 storage using PR #63's 16 × 16 traversal.
- Fused imputation and standard scaling, one row at a time, writing directly into the packed output.
- The same fused operations with a tile of at most 2,048 Doubles for supported widths. Wider and shorter batches use one row.

All paths fit the existing public imputer and scaler on training data. Held-out transformation uses those fitted parameters. They use the same row-width vDSP addition and array-backed division. The experiment does not replace division with multiplication by a reciprocal or introduce another precision policy.

Input storage remains shared and immutable. Output allocation is included in timing; fitting, fixture creation, and output verification are excluded. The final result is a flat array in input row order. The caller retains column names and original row indices with its input. There is no new transport or ownership object in this trial.

## Validation and bounds

[FusedPreprocessingChecks.swift](../Worker/FusedPreprocessingChecks.swift) checks exact numeric equality, signed zero, and matching NaN positions for Double, Float32, and Float16 outputs. It covers dense and missing inputs, row counts around tile boundaries, odd widths, wide rows, empty held-out inputs, all-missing and constant training columns, nonfinite conversion, repeated/permuted selected rows, immutable input snapshots, and rejected column reordering. The checks preserve the staged API's behavior for infinity and NaN; they do not assert that a model accepts those values.

The timing workload is deterministic synthetic data with separate training and held-out values. The largest input is 65,536 × 64 or 8,192 × 512 Doubles. Execution is serial, with two warmups and nine interleaved samples per mode and a three-minute process timeout. The runner kills its child process group on exit. No model is loaded.

## Run

Build `SwiftSciBenchmarkWorker` in Release for arm64 with the project's pinned dependencies, then run:

```sh
python3 Benchmarks/FusedPreprocessingTrial/run.py \
  --worker /absolute/path/to/SwiftSciBenchmarkWorker \
  --output /absolute/path/to/new-results-directory
```

The runner saves raw timings, executable and source fingerprints, a worker log, and a Markdown comparison. Keep machine-specific results outside the contribution branch.

A speedup here supports further integration testing. It does not establish an inference speedup, Neural Engine placement, cache utilization, physical memory savings, or a production dispatch threshold. Next validation must include the existing prepared Core ML workflow, real held-out data, uniquely owned input, and bounded concurrent callers.
