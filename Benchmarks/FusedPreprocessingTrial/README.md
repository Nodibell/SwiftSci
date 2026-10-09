# Fused preprocessing trial

This experiment asks whether fitted imputation, standard scaling, and row-major packing benefit from sharing a bounded pass over compact columns. A package-only fitted plan shares the fused implementation between the benchmark worker and the Core ML trial. Package-only adapters compare copied and directly filled model inputs. No public preprocessing API is added.

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

A speedup here supports further integration testing. It does not establish an inference speedup, Neural Engine placement, cache utilization, physical memory savings, or a production dispatch threshold. The Core ML extension below measures real held-out data and bounded concurrent callers. The direct-storage extension also covers uniquely owned inputs. Other datasets, model shapes, and model types remain separate experiments.

## Covertype and Core ML workflow

`prepare_covertype.py` verifies the recorded source and held-out selection hashes, then extracts the original 11,340 training rows and 8,192 held-out rows. It does not retrain the frozen 54 → 512 → 512 → 7 classifier. The worker fits Swift imputation and standard scaling on the training partition only.

`run_coreml.py` compares the following staged and copied paths through the existing matrix pool:

- Native prepared imputation and consuming scaling, followed by the public pool preparation API.
- Staged preprocessing and packing, followed by the package-only packed-input adapter.
- Fused preprocessing and packing, followed by the same trial adapter.

The trial adapter accepts finite Float16 arrays in the exact model shape and column order. It reserves storage before allocating, copies the values into private storage, and preserves the source batch's row identity. Changing the caller's array cannot change retained input. Both packed paths pay this extra copy, so the experiment can distinguish fusion from differences in preparation APIs. The adapter is experimental and is not a proposed public API.

The runner starts a separate bounded process for each combination of 1,024 or 8,192 rows, CPU-only or CPU-and-Neural-Engine policy, and one or four callers. Four callers share two prediction slots. Each process has a four-minute timeout. The worker checks sampled residency against 1 GiB, verifies every prediction against the native path under the same policy, and requires all owners and reservations to drain. Sampled residency is not a peak-memory limit. Input arrays and returned predictions remain caller-owned outside the pool's reservation.

```sh
python3 Benchmarks/FusedPreprocessingTrial/prepare_covertype.py \
  --existing /absolute/path/to/recorded-covertype-run \
  --output /absolute/path/to/new-raw-covertype.json

python3 Benchmarks/FusedPreprocessingTrial/run_coreml.py \
  --worker /absolute/path/to/SwiftSciBenchmarkWorker \
  --prepared /absolute/path/to/recorded-covertype-run/prepared \
  --models /absolute/path/to/recorded-program-models \
  --raw /absolute/path/to/new-raw-covertype.json \
  --output /absolute/path/to/new-workflow-results
```

The model directory must contain `models-1024/program16.mlpackage` and `models-8192/program16.mlpackage` for the recorded width-512 classifier. `--smoke` runs only the 1,024-row CPU case with one caller. The report separates wall time from preparation and prediction samples. Prediction equivalence under a compute policy does not establish equivalence to Double arithmetic or certify placement on the Neural Engine.

## Direct storage and unique ownership experiment

The direct mode uses an immutable `CoreMLTrialInputContract` containing shape, type, and column names. The contract holds no model or pool reference. Caller tasks reserve input capacity, allocate private storage, and apply the fitted `TrialFusedPreprocessingPlan` directly to it. This avoids the temporary Float16 array and its copy while preserving finite-value checks and row identity. Preparation runs outside the pool actor, so it does not serialize concurrent preprocessing. Each prediction still copies the prepared values into an exclusive pool slot.

The one-caller runs also include `native-owned`. Before timing, the worker constructs genuinely unique columns, then consumes them through the native imputer and scaler. A test verifies that this path reuses the same column addresses. Shared paths retain their source snapshot. Both ownership modes exclude fixture creation, and both use the same single-caller dispatch structure.

The direct and copied fused paths use the same package implementation, fitted values, and vDSP operations. The report presents copied-to-direct ratios separately from native-to-fused results. It also compares direct fusion with the consuming native reference. Those comparisons determine whether removing a copy helps after the native path can already reuse its storage.

Tests cover exact prediction equivalence with missing inputs, concurrent owners, row identity, Float16 overflow rejection, and final reservation release after failure or cancellation. Float16 buffer access remains guarded for macOS 15 and later. The library's deployment minimum is unchanged.

## Specialization at module boundaries

The measured Float16 and Float32 paths call concrete package entry points so the compiler can specialize conversion and array access inside the module. Generic implementations remain available for cross-checking numeric behavior across dtypes. The arithmetic implementation is shared. Check specialization and numeric equivalence when adding another model dtype before using its timings to choose an interface.
