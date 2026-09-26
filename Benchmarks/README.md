# Standardized benchmarks

Use `Tools/bench.py` for reproducible dataframe and numerical comparisons. It verifies input hashes, checks complete outputs against independent answers, runs each case in a fresh process, and preserves every timing sample. Benchmark code is development tooling and is not part of any public library product.

## Run locally

Use an Apple silicon Mac with Xcode selected and Python 3.12 or newer. Run these commands from the repository root. The initial dependency installation and Swift package resolution require network access; prepared datasets require none.

```bash
python3 -m venv Benchmarks/.venv-standardized
Benchmarks/.venv-standardized/bin/python -m pip install -r Benchmarks/Python/requirements-standardized.txt
python3 -m unittest discover -s Benchmarks/Tests
python3 Benchmarks/Tools/bench.py prepare --profile smoke
python3 Benchmarks/Tools/bench.py build
```

The build command prints the worker path. By default it is:

```bash
worker="$HOME/Library/Caches/SwiftSci/standardized-benchmarks/derived/Build/Products/Release/SwiftSciBenchmarkWorker"
python3 Benchmarks/Tools/bench.py run --profile smoke \
  --engines swiftsci,pandas --swift-worker "$worker" \
  --python Benchmarks/.venv-standardized/bin/python \
  --output Benchmarks/Runs/smoke-01
python3 Benchmarks/Tools/bench.py audit Benchmarks/Runs/smoke-01
python3 Benchmarks/Tools/bench.py report Benchmarks/Runs/smoke-01
```

Choose a new output directory for every run. Existing evidence is never overwritten. Rebuild after changing sources or after another Xcode command replaces the worker in the same derived-data directory. A binary checksum mismatch requires a rebuild; do not edit the build record. The builder copies Git-tracked files, including staged additions and their working contents, into a separate cache directory. Stage new source files before building. This excludes untracked cloud-sync duplicate files without changing the checkout. Xcode builds Release with coverage disabled and the dependencies pinned in `Package.resolved`. It permits package plugins for this invocation only.

## Profiles and supported operations

| Profile | Table rows | Warmups / measured samples per process | Independent processes per case and engine |
| --- | ---: | ---: | ---: |
| `smoke` | 129 | 0 / 1 | 1 |
| `standard` | 100,000 | 2 / 5 | 3 |
| `extended` | 1,000,000 | 2 / 5 | 3 |
| `certification` | NIST NumAcc4 only | 0 / 1 | 1 |

The first three profiles each contain 11 table workloads and three NIST checks. Both adapters support CSV read, numeric filtering, stable sorting, grouped sum, matrix export, target-vector export, standard scaling, min-max scaling, mean, sample variance and sample standard deviation. The `pandas` adapter uses pandas for dataframe work and NumPy for numerical work.

The table generator has fixed integer formulas, exact quarter fractions, fixed column order and line endings. Each size has a committed SHA-256 and byte count. `prepare` fails if an existing cached file is corrupt. It never silently replaces bad data. NIST NumAcc4 is a small, unchanged reference fixture with published mean and standard deviation; its variance reference is derived from that standard deviation.

These profiles currently cover finite numeric data and one grouping key. `extended` increases size; it does not yet add nulls, text, skewed distributions, H2O, SciPy or Kiraa. Unsupported engine names fail explicitly. Historical Kiraa comparisons remain under `Results/DataFrameOptimization`. They do not become certified by this tooling.

To run the larger profile, prepare it first and change `--profile` and the output directory. Close competing CPU/GPU workloads and keep power conditions consistent before taking performance measurements. The controller serializes its own builds and runs within this checkout; it cannot prevent other applications or checkouts from consuming resources.

## What is measured

Each workload specification declares its timing boundary. CSV measures verified, warm-cache file parsing into a materialized frame. Filter, sort and group operations start with a prepared frame. Matrix/vector export includes conversion. Scaling includes fitting, transformation and exporting the result. Numerical reductions start with a prepared vector. Input preparation, reference evaluation and output validation are outside the timer.

Both workers retain the result through the end timestamp, then validate every output element. Sorting must preserve input order for ties. Group results are converted to numeric keys and ordered outside timing so that different native result representations can be compared. Exact workloads require exact numeric values; numerical workloads declare absolute and relative tolerances in `Specs/workloads`.

`peak_rss_bytes` is the whole worker process lifetime high-water mark, including imports, setup and validation. It is not operation allocation or logical dataframe storage. BLAS/OpenMP thread environment variables are set to one; this does not limit Swift task concurrency or every library's internal threads.

The reporter takes the median within each process, then the median of those process medians. It retains all samples and does not trim outliers. Durations below one microsecond are marked unresolved for speedup reporting. This conservative threshold is not a measured clock-resolution guarantee. Smoke and certification timings are informational.

## Certification and comparisons

```bash
python3 Benchmarks/Tools/bench.py certify --engines swiftsci,pandas \
  --swift-worker "$worker" --python Benchmarks/.venv-standardized/bin/python \
  --output Benchmarks/Runs/certification-01
python3 Benchmarks/Tools/bench.py audit Benchmarks/Runs/certification-01
python3 Benchmarks/Tools/bench.py compare Benchmarks/Runs/baseline Benchmarks/Runs/candidate
```

A certificate is a local workload-conformance record. It binds the run checksum, experiment contract, source fingerprint, case identities and validated sample count. It is unsigned and does not imply NIST endorsement, third-party accreditation, or correctness of untested APIs. `audit` checks the checksum, complete case coverage, sample counts and validation status. Crashes, timeouts, missing responses and output mismatches fail the run and cannot yield a passing certificate.

`compare` takes existing run directories, not Git revisions. Build and run each revision with the same benchmark tooling and profile on the same machine. Comparisons reject different experiment contracts, environments, reference engine versions or case coverage. Inspect the recorded dependency locks and build commands before attributing a difference solely to library code. The command reports each engine's before/after ratio; it does not claim statistical significance or apply a performance gate.

`run.json` retains the resolved plan, source-file hashes, binary identity, build flags, dependency-lock hash, environment, raw samples and summaries. Per-process requests, responses and logs remain beside it. `events.jsonl` and incremental run records preserve completed work if interrupted; an interrupted run cannot pass an audit.

## CI and development

The `Benchmark conformance` workflow runs protocol tests and both adapters on the smoke profile, then audits the result. It retains evidence for 30 days, including failed runs. Shared-runner timings do not gate a PR. The normal package test suite includes `SwiftSciBenchmarkSupportTests`, which rejects invalid output, corrupt data and malformed binary fixtures.

Add a workload by defining its semantics and tolerance, adding an independent reference and both worker implementations, then including it in a profile. Change the workload version when semantics change. Add generator versions and checksum manifests for new datasets. A faster result is usable only after its output passes validation.

The older `SwiftSciBenchmarks`, `Python/benchmarks.py` and `Python/compare.py` cover additional APIs. Their outputs use a legacy format with different input and measurement rules. Preserve them for historical reproduction; use the standardized runner for claims about the migrated operations. The supported coverage and current limits are listed above.
