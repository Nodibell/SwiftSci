# Bounded inference benchmark

This benchmark compares the existing serial Core ML API with `CoreMLBatchPipeline` using one, two, or four admitted batches. Prediction concurrency is one or two slots. The same training-fitted preprocessing, frozen classifier, batch boundaries, and output checks apply to every variant.

## Workload and timing

The fixture contains 11,340 training rows and 8,192 held-out Covertype rows. The model is the previously qualified 54 → 512 → 512 → 7 Float16 classifier. Fixture conversion writes column-major Double batches to a temporary file, then releases the decoded dataset before timing. Each timed source reads and decodes one batch at a time.

A run processes eight batches of either 1,024 or 8,192 rows. The larger case repeats the recorded held-out population eight times. This exercises sustained processing without claiming additional independent observations. Consumer delays of zero and five milliseconds test a fast sink and backpressure.

Timing includes compiled-model loading, file reads, decoding, preparation, prediction, output hashing, optional consumer delay, and shutdown. It excludes training fit, fixture conversion, and model compilation. The temporary file may be in the operating system's file cache. This is not a cold-disk bandwidth benchmark.

The runner reports medians of three interleaved samples after one warmup per variant. It records first-result latency, total throughput, maximum outstanding batches, sampled resident memory, reservation peaks, thermal state, and source and executable fingerprints. Sampled residency is not a peak-allocation measurement. Each case runs in its own process with a four-minute timeout and a sampled 1 GiB residency cutoff.

The serial reference calls the existing public preparation and prediction APIs and waits for each input reservation to drain. It budgets the pool and preparation separately. The pipeline reserves its complete window upfront, so the two reservation peaks measure different accounting scopes.

## Run the comparison

Build `SwiftSciBenchmarkWorker` in Release for arm64 with coverage instrumentation disabled and the benchmark suite's `.build.json` record. The runner verifies the executable, source tree, Metal resources, and toolchain against that record before timing. Use the existing [Covertype fixture preparation](../FusedPreprocessingTrial/README.md#covertype-and-core-ml-workflow) to obtain the verified raw fixture and frozen model packages.

```sh
python3 Benchmarks/BoundedInference/run.py \
  --worker /absolute/path/to/SwiftSciBenchmarkWorker \
  --models /absolute/path/to/model-directory \
  --raw /absolute/path/to/raw-covertype.json \
  --output /absolute/path/to/new-results-directory
```

The model directory contains `models-1024/program16.mlpackage` and `models-8192/program16.mlpackage`. Add `--smoke` to run only the 1,024-row CPU case without consumer delay. Keep machine-specific results outside the contribution branch.

## Interpret results

Every output hash must match the serial reference under the same compute policy. Every result must preserve local row identity and source order. The worker rejects an exceeded batch window, an incomplete delivery, or a reservation that remains after shutdown. The unit tests separately cover blocked consumers, admission cancellation, competing runs, source errors, invalid shapes, nonfinite conversion, and consumer failure.

CPU-and-Neural-Engine policy permits both devices. It does not establish Neural Engine placement or Double-precision equivalence. A larger window can trade retained memory and first-result latency for throughput. Use the measured application workload to choose the window; these results do not define an automatic hardware dispatch threshold.

## Run sustained load and shutdown checks

Add `--sustained` to the same command. Each window configuration runs in a fresh process. Fast-consumer cases process 1,024 batches per sample; delayed-consumer cases process 128. Each process computes the serial reference, warms its selected variant, and records three samples.

The file contains at most one 8,192-row source cycle. The reader seeks within that cycle and verifies every later prediction against the corresponding first-cycle hash. Fixture storage, retained hashes, and memory checkpoints remain bounded as the number of processed batches increases. The repeated data tests sustained processing, not a larger independent dataset.

Steady-state time starts after 32 completed batches and ends after the final consumer completes. It includes file decoding, preprocessing, prediction, output hashing, and consumer delay. The report separates model-ready time at the first source callback from steady-state throughput. Compiled-model and operating-system caches may already be warm.

The memory report includes the second-half RSS range and end-to-start change for every configuration. Raw checkpoints accompany the report. Framework caches and allocator retention affect RSS, so a flat trace supports bounded-run behavior without proving leak freedom. No growth threshold substitutes for inspection of the trace.

Separate real-model runs inject source failure, consumer failure, source cancellation, and consumer cancellation after 32 batches. Every case checks that the full reservation exists at the trigger, reaches zero before completion, and can be acquired again. Cancellation occurs at callback boundaries while the pipeline is active. The existing pool shutdown tests cover retaining ownership until admitted operations finish; these measurements do not prove device-command interruption.

The sustained runner first checks the actual file adapter in isolation on Swift's cooperative executor. Foundation read temporaries must drain after each synchronous decode. Without that boundary, the regression retains hundreds of MiB across 128 reads even when numeric batches do not escape. The adapter uses `autoreleasepool` around reading and decoding; the returned Swift columns retain their own values. The 64 MiB growth cutoff applies to this fixed, model-free source regression. Core ML runs retain the separate 1 GiB process cutoff and report their memory traces without a leak-freedom claim.
