# Mixed-workload qualification

This experiment compares three public-API workloads under one shared `MemoryBudget`. It changes benchmark code only.

- CPU imputation and standard scaling use training-fitted `Imputer` and `StandardScaler` on prepared Covertype columns.
- GPU regression uses `LinearRegression` with explicit GPU routing and fixed, exactly representable parameters. Predictions return materialized values before admission ends.
- Core ML inference uses the frozen Covertype classifier through `CoreMLBatchPipeline`, with fitted preprocessing and a two-batch window.

Each lane processes the same number of batches. They are different operations, so compare the time to complete the whole work set. Do not interpret their combined row count as one interchangeable prediction rate.

## Run the comparison

Build an uninstrumented Release `SwiftSciBenchmarkWorker` with the existing [benchmark build recorder](../Tools/runner.py). The controller requires matching source, executable, Metal-resource, and toolchain fingerprints.

```sh
python3 Benchmarks/MixedWorkloads/run.py \
  --worker /absolute/path/SwiftSciBenchmarkWorker \
  --models /absolute/path/coreml-models \
  --raw /absolute/path/raw-covertype.json \
  --output /absolute/path/new-results-directory
```

Use `--smoke` for one 32-batch case. The full matrix uses 256 batches per lane, 1,024 and 8,192 rows, CPU-only and CPU-and-Neural-Engine Core ML policies, and admission groups of eight or 256 batches. The model directory contains `models-1024/program16.mlpackage` and `models-8192/program16.mlpackage`. Reuse the frozen artifacts described in [the bounded inference benchmark](../BoundedInference/README.md).

Each configuration runs in a fresh process with a six-minute timeout. Processes run sequentially. The worker rejects sampled process residency at or above 1 GiB. Keep other demanding work idle for a reproducible comparison.

Run the report controls with:

```sh
python3 -m unittest discover -s Benchmarks/Tests -p 'test_mixed_workloads.py'
```

## Read the measurements

Each process runs one isolated warmup, then three rotated samples of four modes. Isolated runs execute the lanes sequentially. Tight admission allows one lane at a time. Balanced admission permits some overlap. Wide admission fits all three lanes.

CPU and GPU reserve 64 MiB per batch. Core ML reserves 113 MiB for each inference run, including its two-batch window. The shared limits are 113, 177, and 241 MiB. These are conservative experiment allowances, not measured allocation sizes or recommended universal defaults.

The Core ML lane reacquires admission and loads its model for each group of batches. Eight-batch groups provide more opportunities for other callers to progress but pay more model-loading overhead. A whole-run reservation minimizes reloads but can delay other lanes. This is an application-level comparison, not a proposed change to the production FIFO queue.

The report separates shared-budget waiting from admitted service time. Service includes output verification. Completion gaps include the first admission and describe chunks for Core ML. Fixed inputs, fitted state, fixture conversion, and model compilation are outside timing. The retained fixture is outside the shared operation budget and remains visible in sampled RSS. Model caches and allocator retention remain across samples.

## Check correctness and shutdown

CPU results must match the public row-array path before timing, then match their isolated hashes on every iteration. GPU predictions must satisfy an independent Float rounding bound before timing, then match their isolated hashes exactly. Core ML hashes must match isolated inference under the same compute policy. This checks contention effects without changing numerical tolerances or claiming Double accuracy for Float arithmetic.

One test cancels the queued Core ML lane behind a full-budget blocker while CPU and GPU peers wait. Another cancels inference after 16 consumed batches while the other lanes continue. All admitted work must drain, both peers must finish, and the complete budget must be reacquired afterward. Cancellation does not establish that a native device instruction was interrupted.

The worker records queue depth, waits, service durations, completion times, sampled residency, thermal state, and reservation peaks. Finite completion demonstrates progress for these workloads. It does not establish unlimited-time starvation freedom, exclusive Neural Engine placement, leak freedom, or cache utilization.
