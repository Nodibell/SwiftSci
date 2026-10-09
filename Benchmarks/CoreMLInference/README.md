# Core ML inference qualification

This qualification exercises `CoreMLPredictor` and compares it with native SwiftSci
CPU and GPU prediction. It is separate from the standardized production profiles.

Build the existing benchmark worker from the committed branch:

```sh
python3 Benchmarks/Tools/bench.py build \
  --cache "$HOME/Library/Caches/SwiftSci/coreml-qualification" \
  --packages "$HOME/Library/Caches/SwiftSci/packages"
```

Use the worker path returned by that command. Choose a new output directory:

```sh
python3 Benchmarks/CoreMLInference/run.py \
  --worker /absolute/path/to/SwiftSciBenchmarkWorker \
  --output /absolute/path/to/new-coreml-results
```

The controller launches six fresh processes, with a 120-second limit per mode and
process-group cleanup. It stops on a runtime error, malformed evidence, or timeout.
Each process evaluates a deterministic 256-row, 128-feature linear model with the same binary-exact weights and input values.
A Double scalar calculation supplies the expected output. Every prediction is checked
with absolute and relative tolerances of `1e-5`.

The native comparisons use public `LinearRegression` with supplied weights on explicit
CPU and GPU paths. Core ML uses the existing neural exporter with one dense layer and
CPU-only, CPU/GPU, CPU/Neural Engine, and all-device policies. This isolates the initial
inference connection; it does not represent all neural architectures or precisions.

Each mode records:

- Model loading time, separately from model compilation.
- First request and 32 measured requests after three additional warmups (use
  `--samples 256` for a longer memory check, up to 512 samples).
- Complete-output validation and output hashes using the suite's `BenchmarkSample`.
- Adapter copy counters and reservation cleanup for Core ML.
- Resident memory after loading, warmup, and each measured request, plus process peak RSS.
- Anticipated layer placement from `MLComputePlan` on macOS 14.4 or later.

`REPORT.md` presents timing, memory, and agreement with the Double reference for every
completed case. `assessment.json` separates execution completion, reference accuracy,
and application suitability. JSON records preserve the measurements and source,
worker, model, and input fingerprints. The controller requires the standard builder's `.build.json` and checks the worker,
source, and Metal fingerprints before running. It rejects coverage-instrumented binaries
and changes to source files during qualification.

The default comparison submits one row at a time. The native predictors process the entire
matrix per call. Timing includes the adapter and result extraction, with validation
outside the timed interval. A Core ML slowdown here can reflect call and conversion
costs; it does not establish inferior hardware throughput. Model loading may reuse
Core ML's on-disk cache, so a fresh process does not guarantee an uncached load.

A compute plan is not an execution trace. Use Instruments to establish actual Neural
Engine execution. RSS includes runtime caches and is not a leak detector. The budget
covers estimated participating allocations, not total process memory. No result from
this qualification constitutes a production performance baseline or numerical certification.

## Batch and model-complexity sweep

Add `--sweep --samples 16` to run the bounded comparison matrix. The sweep uses
1, 32, and 256 observations and these fixed-weight models:

| Workload | Dense layer widths | Hidden activation |
|---|---|---|
| `linear128` | 128 → 1 | None |
| `mlp128` | 128 → 128 → 128 → 1 | ReLU |
| `mlp512` | 512 → 512 → 512 → 1 | ReLU |

The sweep compares one-example calls with batches of up to 32 examples through the
public adapter. Each batch retains distinct input vectors until prediction completes.
It also includes fixed-shape matrix models through the public adapter and a direct
Core ML reference. Their rank-two inputs feed all observations to one prediction call,
using the same weights and biases as the vector models. Their outputs are checked against the same scalar Double calculation.
No training or reference calculation occurs inside a timed interval.

Matrix models use the documented rank-two
[inner-product semantics](https://apple.github.io/coremltools/mlmodel/Format/NeuralNetwork.html#innerproductlayerparams).
The `-matrix-adapter` modes use the public `inputLayout: .matrix` contract. The
`-matrix` modes call Core ML directly using the identical artifact, to measure adapter
overhead. Both remain controlled research fixtures rather than trained production models.
For matrix modes, the table's Maximum batch column is the number of matrix rows; the
public API's `maximumBatchSize` stays at 1.
Core ML's [batch prediction API](https://developer.apple.com/documentation/coreml/mlmodel/predictions(from:options:))
accepts multiple examples but does not promise a fused matrix operation.

The native `LinearRegression` comparisons run only for `linear128`. The `mlx-cpu`
and `mlx-gpu` paths use a benchmark-only fixed-weight Float32 matrix implementation
for all three models. They pack the entire request and materialize every output.
They do not measure SwiftSci's public MLP estimators. Core ML input transport is Double,
with Float32 weights and device-selected internal arithmetic. Tolerances stay at
`1e-5` absolute and relative on every path.

Every cell uses a fresh process, with a 120-second timeout and process-group cleanup.
The worker validates each request, including warmups. The controller checks parameter
and input hashes across paths and model hashes within each calling shape. A result
outside the reference tolerance remains an accuracy exceedance while the sweep
measures the remaining cases. Its timing stays visible beside its error. Exceeding
a reference requirement alone does not establish an integration or hardware defect.
A runtime error or timeout stops the sweep and preserves its log. The report includes
first-request timing, warm medians, rows per second, resident memory, numerical error,
and planned devices.
The small deterministic models isolate execution costs; they do not represent trained
model quality, transformer inference, or a production performance baseline.

Use `--sweep --rows 1024` for the larger-batch extension, or supply several row counts
between 1 and 1024. The default sweep remains 1, 32, and 256. Workloads run serially;
do not run other benchmark builds concurrently with these measurements.

Run the controller contract checks without a GPU:

```sh
python3 -m unittest discover -s Benchmarks/CoreMLInference -p 'test_*.py'
```

## Interpret completion and accuracy

The comparison records three separate assessments:

- Execution completion means the worker returned complete, finite outputs with valid
  measurement records and released its admission reservation. It does not certify
  every aspect of integration correctness.
- Reference accuracy means every checked prediction, including warmups, met
  `abs(actual - expected) <= 1e-5 + 1e-5 * abs(expected)` against the independent
  scalar Double calculation. The report labels this `Within tolerance` or
  `Outside tolerance` for every device policy.
- Application suitability is `not_assessed`. These fixed-weight fixtures do not
  define acceptable quality for a trained model or scientific application.

Reduced-precision execution can complete correctly and exceed the Double-reference
requirement. Report its measured error alongside performance. Select an application
accuracy contract before claiming that paths provide interchangeable results. This
suite does not invent a looser tolerance based on the selected device, or treat a
fit to observed hardware arithmetic as proof of application quality.

A completed observational run exits 0. Use `--require-reference-accuracy` when the
Double-reference requirement must gate a check:

```sh
python3 Benchmarks/CoreMLInference/run.py \
  --worker /absolute/path/to/SwiftSciBenchmarkWorker \
  --output /absolute/path/to/new-strict-results \
  --sweep --require-reference-accuracy
```

That option exits 2 if any case exceeds the unchanged reference tolerance. Execution
and evidence errors remain errors in either mode. Existing standardized scientific
certification profiles are unchanged.

Raw worker records retain the legacy names `validated`, `mismatchedValues`, and
`warmupFailures` for compatibility. They describe reference-accuracy checks, not
hardware health. Worker exit 2 carries that accuracy result; the controller separates
it from execution failure and applies the requested gate. Existing result files and
benchmark fingerprints do not need rewriting.

## Trained models on frozen real-data splits

`trained.py` compares a two-hidden-layer ReLU network on the repository's WDBC
classification and red-wine regression fixtures. It requires NumPy, SciPy,
scikit-learn, and threadpoolctl in the benchmark Python environment.

```sh
Benchmarks/.venv-standardized/bin/python Benchmarks/CoreMLInference/trained.py \
  --worker /absolute/path/to/SwiftSciBenchmarkWorker \
  --output /absolute/path/to/new-trained-results --samples 64
```

The controller verifies the fixture checksums and uses the existing disjoint
train/validation/test partitions. It fits scaling and model parameters on training
rows only, with a fixed seed and configuration. Weights are rounded once to
Float32, then shared by every execution path. The original fitted model's scores,
rounded model's scores, training warnings, and constant-predictor scores are saved.
This is a fixed model comparison, not hyperparameter selection or a quality certificate.

The 66 cases cover one row, 32 rows, and the complete held-out test partition.
They compare MLX CPU/GPU, Core ML example batches, direct matrix calls, and public
matrix calls under each compute policy. Each case runs in a fresh process with
three additional warmups and the requested measured samples. Complete output hashes
must remain stable before the first output is used for held-out task scoring.
Public and direct matrix outputs must match. Numerical accuracy checks retain the
existing Double-reference tolerance and do not replace task scores.

Classification outputs are logits. The controller applies sigmoid and reports
accuracy, ROC-AUC, log loss, and label disagreements. Regression outputs are in
training-standardized target units; the controller reverses that scaling before
reporting RMSE, MAE, and R-squared. Numerical differences are measured in the
model's output units. Task scoring is outside timed inference.

`results.json`, frozen model fixtures, fitted preprocessing, split identities,
`provenance.json`, and `requirements.txt` retain the evidence. Python estimator
inference and scaling-plus-inference timings are recorded separately. Classification
estimator timing includes probability conversion, while the Swift workers return
logits; these timings do not isolate identical arithmetic. Core ML input packing,
cooperative admission, and owned output extraction are included in worker timing.
Training and preparation of the already-standardized input batch are outside it.

The [supervised fixtures](../Fixtures/supervised/README.md) separately exercise
SwiftSci's training and preprocessing APIs against independent references. The
trained Core ML run evaluates frozen-model transport and inference, not SwiftSci
neural-model training. Published held-out partitions are development benchmarks,
not fresh final evaluation sets. These runs are not official MLPerf results.

## Covertype multiclass stress sweep

`covertype.py` uses the original Covertype partition: the first 11,340 observations
for training, the next 3,780 for validation, and the remaining 565,892 for testing.
The data is from Jock Blackard, [Covertype, UCI](https://doi.org/10.24432/C50K5N),
licensed CC BY 4.0. Keep its archive and `covtype.info` with the run evidence.

Download the archive from UCI's dataset page and extract `covtype.data.gz` and
`covtype.info` into `<output>/data`. Then run:

```sh
Benchmarks/.venv-standardized/bin/python Benchmarks/CoreMLInference/covertype.py \
  prepare --output /absolute/path/to/covertype-results
Benchmarks/.venv-standardized/bin/python Benchmarks/CoreMLInference/covertype.py \
  run --output /absolute/path/to/covertype-results \
  --worker /absolute/path/to/SwiftSciBenchmarkWorker
```

Preparation expects the original archive at `<output>/data/covertype.zip` and
records its hash. It checks dataset dimensions and labels, fits scaling on the
training partition, and trains three ReLU classifiers with two hidden layers of
width 128, 256, or 512. Adam uses a fixed seed and settings. Early stopping uses a
validation subset drawn only from the training partition. The original validation
and test partitions do not control model fitting. Model size is an experimental
factor; these runs do not select a model using test scores.

The Python reference evaluates the complete test partition in bounded chunks.
A fixed, randomly selected set of 8,192 test observations supplies the inference
sweep. Each runtime receives the same features and once-rounded Float32 weights.
The five request sizes are 1, 32, 256, 1,024, and 8,192 rows. All seven output
logits are checked before scoring accuracy, macro-F1, log loss, per-class recall,
and disagreement with the reference labels. Full-partition Python quality and
sampled runtime quality are separate measurements.

Inputs and expected logits are row-major little-endian Float64 binary files.
Their dimensions and SHA-256 hashes are recorded in the small JSON fixture
metadata. The worker rejects wrong sizes, checksums, shapes, and nonfinite values.
It also accepts the earlier single-output JSON fixtures. Binary inputs still
require preparation into owned columns; this path is not a zero-copy claim.

The sweep runs 90 fresh-process cases serially with 16 measured predictions after
warmup. Core ML workers use a 256 MiB cooperative admission budget and a 120-second
process timeout. Framework allocations and model storage remain outside that
budget. This is a bounded capacity sweep, not a system-memory saturation test.
The controller preserves numerical tolerance exceedances as accuracy evidence and
requires stable prediction hashes before scoring the task outputs.

The fixture loader and matrix benchmark support up to 8,192 rows and 1,024 output
columns; the production adapter's contract is unchanged. Concurrency, queue
latency, sustained-load duration, and the complete read-to-inference pipeline need
separate runs. A favorable compute plan alone does not prove hardware execution.

## Concurrent callers and sustained load

After preparing the Covertype fixtures with `covertype.py`, run:

```sh
python3 Benchmarks/CoreMLInference/concurrency.py \
  --worker /absolute/path/to/SwiftSciBenchmarkWorker \
  --prepared /absolute/path/to/covertype/prepared \
  --output /absolute/path/to/new-concurrency-results
```

The controller verifies the worker and fixture fingerprints before starting. It runs
fresh processes serially, using the frozen 54 → 512 → 512 → 7 classifier and 8,192 rows.
The comparison covers CPU-only and CPU/Neural Engine policies with:

- One caller and one predictor as the serial baseline.
- Four callers sharing one predictor and one request's admission capacity.
- Four callers with separate predictors, sharing one request's admission capacity.
- Four callers with separate predictors, sharing two requests' admission capacity.

Each comparison lasts 15 seconds. Two further runs use four predictors and two admission
slots for five minutes per policy. Use `--sweep-seconds` to choose 1–60 seconds and
`--sustain-seconds` to choose 0–1,800 seconds per policy. Zero disables the sustained runs.
The controller applies a process timeout of the requested duration plus 120 seconds.
An observer checks RSS once per second and stops the run if it exceeds 1 GiB.
This sampled cutoff does not prevent a short allocation spike between checks.

Each process first holds the budget closed, queues four requests, cancels two, and
verifies that the remaining two finish and admission drains. It also verifies cleanup
when a request contains a nonfinite value. These probes use separate admission accounting
from the measured cancellation-free loop.

Timed callers alternate normal and reversed input rows. Every complete output is hashed
and checked against a serial preflight result under the same compute policy. Row identity
must also match. At completion, the harness releases every predictor and verifies that
retained first and last outputs remain valid. It retains only two outputs per caller,
256 latency buckets per caller, and at most 1,802 memory observations per process.

Request latency includes actor scheduling, admission wait, input packing, prediction,
and owned-output construction. The public API does not expose those intervals separately.
Output validation is outside request latency but inside the elapsed run time used for
throughput. Reported percentiles are histogram upper bounds with 8% bucket spacing.
These measurements therefore differ from isolated kernel timings.

The admission budget covers estimated request storage, including an 8 MiB workspace
allowance per request. It does not cap persistent models, retained results, runtime
caches, or total RSS. Preflight disagreement with the strict Double reference remains
in the evidence and is separate from concurrent output repeatability. A stable RSS
series and released Swift owners do not prove that every framework allocation is freed.
The compute plan identifies anticipated placement; execution tracing remains separate.

## Native async trial

The `codex/coreml-memory-integration-trial` branch adds a package-only matrix
execution switch. Public construction keeps synchronous prediction. Both modes
share the same packing, validation, output extraction, and memory estimate.
The async mode transfers an independent provider for every request and uses Core
ML's native async prediction on a shared loaded model.

```sh
python3 Benchmarks/CoreMLInference/async_trial.py \
  --worker /absolute/path/to/SwiftSciBenchmarkWorker \
  --prepared /absolute/path/to/covertype/prepared \
  --output /absolute/path/to/new-async-results
```

The trial compares synchronous serialization, synchronous replicas, and one async
model with one, two, or four admission slots. It repeats the concurrency harness's
cancellation, ownership, row-identity, and complete-output checks. Use `--seconds`
to choose 1–300 seconds per configuration. Model weights and transport precision
remain unchanged; this trial does not test reusable buffers or Float16 transport.

The private async owner documents Core ML's thread-safety guarantee in one
`@unchecked Sendable` wrapper. It exposes only native async prediction. Request
providers cross the Swift concurrency seam through `sending` parameters and
results; they are never reused across simultaneous predictions. The actor still
owns configuration, input packing, and output extraction. No mutable prediction
state is shared between requests.

Use Instruments recordings separately from timing runs. A profiling-only copy
may need the `com.apple.security.get-task-allow` entitlement. Record the copied
binary's signature and fingerprint rather than replacing the measured worker.
A completed recording must still be inspected for actual device events before
claiming Neural Engine execution or compute overlap.

## Compare matrix adapter revisions

`compare_matrix.py` runs paired before/after measurements through the public fixed-matrix adapter. Preserve each Release worker and its build record before rebuilding. The controller checks the worker hashes, requires identical benchmark logic, alternates execution order, and verifies matching model/input/output fingerprints and admission charges. It records thermal state and host load for each process.

```sh
python3 Benchmarks/CoreMLInference/compare_matrix.py \
  --before-worker /path/to/before-worker --before-build /path/to/before-build.json \
  --after-worker /path/to/after-worker --after-build /path/to/after-build.json \
  --fixtures /path/to/prepared-covertype-fixtures \
  --output /path/to/new-comparison --samples 64 --repeats 3
```

The fixture directory must contain the verified Covertype `fixture-512-1024.json` and `fixture-512-8192.json` files and their referenced data. The comparison also includes controlled 128-feature linear and 512-feature multilayer models, from single-row requests to 1,024-row matrices. CPU, GPU, and Neural Engine compute preferences run serially. Each child process has a timeout and process-group cleanup.

Reported speedups compare medians of warm per-process medians. Timing includes adapter admission, packing, prediction, extraction, and the same result flattening on both revisions. Precision exceedances remain separate from execution success. Non-nominal thermal states or competing host work make these exploratory measurements; repeat under controlled conditions before publishing a performance baseline.

## Persistent matrix pool comparison

`pool_trial.py` compares the public synchronous predictor, the package-only fresh
async predictor, and the public `CoreMLMatrixPool`. It uses identical Double matrix
transport and frozen Covertype models at 1,024 and 8,192 rows, with CPU-only and
CPU/Neural Engine policies. Each mode uses one model. The serial case has one caller
and one slot; the concurrent case has four callers and two slots.

```sh
python3 Benchmarks/CoreMLInference/pool_trial.py \
  --worker /absolute/path/to/SwiftSciBenchmarkWorker \
  --prepared /absolute/path/to/covertype/prepared \
  --output /absolute/path/to/new-pool-results --seconds 8
```

The 24 configurations run serially in fresh processes in a fixed shuffled order.
Use `--seconds` to choose 1–300 seconds per configuration. Worker and fixture
fingerprints, output hashes, the 1 GiB sampled RSS cutoff, and process timeouts use
the shared concurrency controller. Coverage or sanitizer instrumentation rejects
the timing worker. The report includes latency, throughput, and sampled RSS.

Persistent mode reserves 32 MiB for retained state plus the same conservative
per-request allowance used by the fresh modes for each slot. It checks that the
whole quota stays reserved between requests and returns to zero after scope exit.
Fresh modes account only for active requests and leave model memory outside the
budget. These policies are deliberately reported separately. Neither estimate
is an allocator-enforced memory limit.

All modes check invalid-input recovery, owner release, row identity, and retained
output independence. Fresh modes also check queued request cancellation in each
process. Persistent cancellation is tested by `CoreMLMatrixPoolAdmissionTests`,
not by this timing run. Repeatable outputs do not establish numerical accuracy,
and the compute policy does not prove actual Neural Engine execution.

### Trained ML Program coverage

Use `--models` to run the same 24 configurations against the Float32-I/O ML Programs
created by [the model preparation tool](prepare_models.py). The directory
must contain `models-1024` and `models-8192`, each with its original `manifest.json`
and `program32.mlpackage`:

```sh
python3 Benchmarks/CoreMLInference/pool_trial.py \
  --worker /absolute/path/to/SwiftSciBenchmarkWorker \
  --prepared /absolute/path/to/covertype/prepared \
  --models /absolute/path/to/transport-models \
  --output /absolute/path/to/new-program-pool-results --seconds 8
```

The controller verifies each manifest against the reference fixture and checks every
recorded model file before and after the run. Each worker compiles the source package
and checks that its fixed input and output shapes match the fixture with Float32 I/O.
The worker records a hash of all source-package files, including weights.

These programs use Float16 internal computation. Float32 transport does not establish
Float32 arithmetic throughout the model. Compare pooled results with fresh predictions
from the same program and compute policy; retain the independent Double-reference
mismatch counts separately.

Package-only pool counters record successful predictions and how often Core ML returns
the supplied output object. The report includes preflight predictions in these counts.
An identity match proves reuse of that object, not the absence of internal copies.
A different object may share storage, which this counter does not detect. The benchmark
accepts either observation when outputs, row identity, and ownership checks pass.

### Allocation and device traces

`profile_pool.py` records six bounded Instruments traces using a completed ML Program
comparison as its reference. It profiles fresh and pooled predictions at 1,024 rows,
with four callers, one model, and two admitted requests. The four allocation traces
cover CPU-only and CPU/Neural Engine policies. Two additional traces capture Core ML
and Neural Engine events under the CPU/Neural Engine policy.

```sh
python3 Benchmarks/CoreMLInference/profile_pool.py \
  --worker /absolute/path/to/SwiftSciBenchmarkWorker \
  --reference /absolute/path/to/completed-program-pool-results \
  --output /absolute/path/to/new-profile-results --seconds 2
```

Use a Release worker with its matching build record. The controller verifies source,
worker, Metal, and model fingerprints and compares output hashes with the reference.
It copies the worker into the output directory and signs that copy with the local
`get-task-allow` entitlement so Instruments can attach. The original worker remains
unchanged. Xcode command-line tools and local profiling permission are required.

Traces run serially. Each worker measures 1–10 seconds of predictions. Recorder
timeouts also allow for launch and trace saving. Cleanup terminates any remaining
recorder and the child matching that run's exact worker and request paths. Allow
several gigabytes of disk space for the traces.

The worker marks `CoreML owner scope` and `CoreML measurement` in Points of Interest.
Use the measurement interval to exclude model loading and preflight checks from event
counts. Allocation Statistics covers the whole trace unless filtered in Instruments.
Object allocation counts exclude separately allocated backing storage, and virtual
address reservations do not measure physical memory consumption.

The compute-plan reader reports preferred devices for neural-network layers and ML
Program operations, including nested blocks. Plans describe intended placement. Use
recorded Neural Engine events to check observed execution separately. Missing device
events do not prove CPU fallback, and buffer identity does not prove zero-copy input.
Profiling changes execution overhead; these runs are not performance comparisons.


## Compare prepared production inputs

`prepared_workflow.py` compares `CoreMLMatrixPool.predict(batch)` with one
`CoreMLMatrixPool.prepare(batch, budget:)` followed by repeated predictions. Both
paths use the production library and the same trained Float16 model, column order,
row selections, compute policy, and slot count.

```sh
python3 Benchmarks/CoreMLInference/prepared_workflow.py \
  --worker /path/to/SwiftSciBenchmarkWorker \
  --models /path/to/coreml-transport-models \
  --fixtures /path/to/prepared-fixtures \
  --output /path/to/new-results
```

The model directory contains `models-1024/program16.mlpackage` and
`models-8192/program16.mlpackage`. Fixtures are the existing trained Covertype
`fixture-512-1024.json` and `fixture-512-8192.json` files with their referenced binary
matrices. Build the worker from the current sources before running. The runner records
worker, source, model, and fixture hashes; it does not rebuild or certify a supplied binary.

Cases alternate direct and retained input in seeded randomized pairs at 1, 2, 4, 16,
and 64 predictions per preparation. Each policy runs with one caller and slot, and with
four callers and two slots. Defaults use three fresh processes and five samples per case.
Totals include preparation and prediction. Model loading, source construction, output
validation, and eventual reservation release are outside timing. Release duration is
recorded separately. Each child process has a timeout; sampled RSS must remain below 1 GiB.

The worker requires identical outputs and row identities across input paths and checks
that input and pool reservations drain. Results retain differences from the strict Double
reference. These checks establish transport equivalence, not application accuracy or
actual Neural Engine placement. `REPORT.md` shows total timing, process ranges, and the
extra retained-input reservation. These are exploratory measurements, not a formal
performance baseline or physical-memory savings.


## Prepare matched ML Program artifacts

Create a separate environment from `requirements-programs.txt`, then run
`prepare_models.py --fixture /path/to/fixture-512-1024.json --output /path/to/models-1024`.
Repeat with the 8,192-row fixture and `models-8192` output directory. The tool verifies
binary fixture hashes, uses the recorded trained weights, and writes matched legacy,
Float32-I/O, and Float16-I/O artifacts with a provenance manifest. Conversion requires
Python and coremltools; using an already prepared model does not.

Standalone IOSurface and custom-buffer experiments remain on the development trial
branch. They are not part of the production adapter or this benchmark suite.
