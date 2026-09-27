# Standardized benchmarks

Use `Tools/bench.py` for reproducible dataframe and numerical comparisons. It verifies input hashes, checks complete outputs against independent answers, runs each case in a fresh process, and preserves every timing sample. Benchmark code is development tooling and is not part of any public library product.

## Run locally

Use an Apple silicon Mac with Xcode selected and Python 3.12 or newer. Run these commands from the repository root. The initial dependency installation and Swift package resolution require network access; prepared datasets require none.

```bash
python3 -m venv Benchmarks/.venv-standardized
Benchmarks/.venv-standardized/bin/python -m pip install -r Benchmarks/Python/requirements-standardized.txt
Benchmarks/.venv-standardized/bin/python -m unittest discover -s Benchmarks/Tests
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py prepare --profile smoke
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py build
```

The build command prints the worker path. By default it is:

```bash
worker="$HOME/Library/Caches/SwiftSci/standardized-benchmarks/derived/Build/Products/Release/SwiftSciBenchmarkWorker"
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py run --profile smoke \
  --engines swiftsci,pandas --swift-worker "$worker" \
  --python Benchmarks/.venv-standardized/bin/python \
  --output Benchmarks/Runs/smoke-01
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py audit Benchmarks/Runs/smoke-01
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py report Benchmarks/Runs/smoke-01
```

Choose a new output directory for every run. Existing evidence is never overwritten. Rebuild after changing sources or after another Xcode command replaces the worker in the same derived-data directory. A binary checksum mismatch requires a rebuild; do not edit the build record. The builder copies Git-tracked files, including staged additions and their working contents, into a separate cache directory. Stage new source files before building. This excludes untracked cloud-sync duplicate files without changing the checkout. Xcode uses the package-level Release `build-for-testing` action with `-enableCodeCoverage NO` and dependencies pinned in `Package.resolved`. This compiles the targets without running the test suite. The generated Swift package scheme can ignore coverage settings during a plain `build` action. The builder and run planner inspect the actual Mach-O binary and reject LLVM profiling or coverage sections. A compiler setting alone is not proof that instrumentation is absent. It permits package plugins for this invocation only.

## Profiles and supported operations

| Profile | Table rows | Warmups / measured samples per process | Independent processes per case and engine |
| --- | ---: | ---: | ---: |
| `smoke` | 129 | 0 / 1 | 1 |
| `standard` | 100,000 | 2 / 5 | 3 |
| `extended` | 1,000,000 | 2 / 5 | 3 |
| `certification` | All nine NIST univariate datasets | 0 / 1 | 1 |
| `nist` | All nine NIST univariate datasets | 2 / 5 | 3 |
| `migration-smoke` | Small numeric, mixed and Parquet fixtures | 0 / 1 | 1 |
| `migration` | Mixed sizes declared per case | 2 / 5 | 3 |
| `public-smoke` | 257 synthetic rows; 1,599 real wine rows | 0 / 1 | 1 |
| `public-data` | 100,000 synthetic rows; 1,599 real wine rows | 2 / 5 | 3 |
| `numerical-conformance` | Eight NIST OLS and eleven ANOVA datasets | 0 / 1 | 1 |
| `numerical-binary64` | Same 19 inputs with exact-binary64 references | 0 / 1 | 1 |
| `model-conformance` | Two PCA and one multinomial Naive Bayes fixture | 0 / 1 | 1 |

The first three profiles each contain 11 table workloads and three NIST checks. Both adapters support CSV read, numeric filtering, stable sorting, grouped sum, matrix export, target-vector export, standard scaling, min-max scaling, mean, sample variance and sample standard deviation. The `pandas` adapter uses pandas for dataframe work and NumPy for numerical work.

The table generator has fixed integer formulas, exact quarter fractions, fixed column order and line endings. Each size has a committed SHA-256 and byte count. `prepare` fails if an existing cached file is corrupt. It never silently replaces bad data. The nine unchanged NIST univariate fixtures cover varied scales and numerical difficulty. Mean and standard deviation use published answers; variance references are derived from those standard deviations. See [NIST coverage and tolerances](Fixtures/nist/README.md).

The original table profiles cover finite numeric data and one grouping key. The certification and NIST profiles each contain 27 numerical cases, using dataset-specific tolerances recorded in the resolved plan. `extended` increases size; it does not yet add nulls, text, skewed distributions, H2O or Kiraa. Unsupported engine names fail explicitly. Historical Kiraa comparisons remain under `Results/DataFrameOptimization`. The Kiraa development build used here is unofficial and has known bugs. It remains an optional experimental reference, never an accuracy oracle or production baseline. It is not a supported standardized engine.

The migration profiles add streaming CSV, mixed numeric/string/Boolean types, joins, Parquet, multi-output grouping, row reductions, correlations, hypothesis tests, regression metrics, ROC-AUC, encoders, SQLite ingestion, RAG summaries and pooling/Dice workloads. Each profile declares 36 cases. Read its resolved plan for the exact input size and operation boundary. The `pandas` adapter also uses SciPy, PyArrow and SQLite where the workload requires them. Parquet interoperability regression tests also cover nulls; the timed migration fixtures do not. These profiles complement the original profiles; they do not replace the NIST checks.

```bash
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py inventory
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py prepare --profile migration-smoke
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py run --profile migration-smoke \
  --engines swiftsci,pandas --swift-worker "$worker" \
  --python Benchmarks/.venv-standardized/bin/python \
  --output Benchmarks/Runs/migration-smoke-01
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py audit Benchmarks/Runs/migration-smoke-01
```

`inventory` reconciles 133 legacy result rows and two diagnostics with standardized workloads or explicit research dispositions. A row marked as migrated identifies its replacement contract; it does not certify historical output. Learned-model and GPU workloads that remain research require `--research` and cannot support certification or production claims.

To run the larger profile, prepare it first and change `--profile` and the output directory. Close competing CPU/GPU workloads and keep power conditions consistent before taking performance measurements. The controller serializes its own builds and runs within this checkout; it cannot prevent other applications or checkouts from consuming resources.

The numerical profiles extend coverage to CPU model fitting and balanced ANOVA. They distinguish original decimal conformance from arithmetic accuracy on binary64 inputs. Difficult cases remain in the profiles even when they fail. Read the [numerical fixture contracts](Fixtures/nist-models/README.md) before interpreting results. These profiles are diagnostic, not performance baselines or a claim of library-wide certification.

The `model-conformance` profile checks exact synthetic PCA and Naive Bayes contracts. Its [fixture documentation](Fixtures/exact-models/README.md) explains sign-invariant comparisons, total-variance ratios and the distinction between class labels and prediction indices. It provides no model-quality claim.

## What is measured

Each workload specification declares its timing boundary. CSV measures verified, warm-cache file parsing into a materialized frame. Filter, sort and group operations start with a prepared frame. Matrix/vector export includes conversion. Scaling includes fitting, transformation and exporting the result. Numerical reductions start with a prepared vector. Input preparation, reference evaluation and output validation are outside the timer. Synchronous operations use a synchronous dispatcher; CSV, Parquet and SQLite operations await their asynchronous APIs. Measurement contract v5 requires an uninstrumented worker and includes streaming chunk checks and independent Parquet validation. It cannot be compared with earlier measurement contracts.

Both workers retain the result through the end timestamp, then validate every output element. Sorting must preserve input order for ties. Group results are converted to numeric keys and ordered outside timing so that different native result representations can be compared. Exact workloads require exact numeric values; numerical workloads declare absolute and relative tolerances in `Specs/workloads`.

`peak_rss_bytes` is the whole worker process lifetime high-water mark, including imports, setup and validation. It is not operation allocation or logical dataframe storage. BLAS/OpenMP thread environment variables are set to one; this does not limit Swift task concurrency or every library's internal threads.

The reporter takes the median within each process, then the median of those process medians. It retains all samples, reports the worst measured absolute output error, and does not trim outliers. Each worker explicitly marks durations below one microsecond as unresolved, including zero when the timer cannot distinguish its start and end. Those samples still undergo complete output validation. The reporter preserves their raw durations and suppresses a speedup ratio if either case contains any unresolved sample. This conservative threshold is not a measured clock-resolution guarantee. Smoke, migration-smoke, public-smoke and certification timings are informational.

## Certification and comparisons

```bash
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py prepare --profile certification
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py certify --engines swiftsci,pandas \
  --swift-worker "$worker" --python Benchmarks/.venv-standardized/bin/python \
  --output Benchmarks/Runs/certification-01
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py audit Benchmarks/Runs/certification-01
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py run --profile nist --engines swiftsci,pandas \
  --swift-worker "$worker" --python Benchmarks/.venv-standardized/bin/python \
  --output Benchmarks/Runs/nist-01
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py audit Benchmarks/Runs/nist-01
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py compare Benchmarks/Runs/baseline Benchmarks/Runs/candidate
```

A certificate is a local workload-conformance record. It binds the run checksum, experiment contract, source fingerprint, case identities and validated sample count. It is unsigned and does not imply NIST endorsement, third-party accreditation, or correctness of untested APIs. `audit` checks the checksum, complete case coverage, sample counts and validation status. Crashes, timeouts, missing responses and output mismatches fail the run and cannot yield a passing certificate.

`compare` takes existing run directories, not Git revisions. Build and run each revision with the same benchmark tooling and profile on the same machine. Comparisons reject different experiment contracts, environments, reference engine versions or case coverage. Inspect the recorded dependency locks and build commands before attributing a difference solely to library code. The command reports each engine's before/after ratio; it does not claim statistical significance or apply a performance gate.

`run.json` retains the resolved plan, source-file hashes, binary identity, build flags, dependency-lock hash, environment, raw samples and summaries. Per-process requests, responses and logs remain beside it. `events.jsonl` and incremental run records preserve completed work if interrupted; an interrupted run cannot pass an audit.

## CI and development

The `Benchmark conformance` workflow runs protocol tests, both adapters on the smoke and migration-smoke profiles, all 27 NIST reference cases per engine, and the public-data smoke cases, then audits the results. It retains evidence for 30 days, including failed runs. Shared-runner timings do not gate a PR. The normal package test suite includes `SwiftSciBenchmarkSupportTests`, which rejects invalid output, corrupt data and malformed binary fixtures.

Add a workload by defining its semantics and tolerance, adding an independent reference and both worker implementations, then including it in a profile. Change the workload version when semantics change. Add generator versions and checksum manifests for new datasets. A faster result is usable only after its output passes validation.

The legacy `SwiftSciBenchmarks`, `Python/benchmarks.py` and `Python/accuracy_benchmarks.py` require `--research`. They label console and JSON output as unvalidated research. Their input and measurement rules differ from the standardized contracts. `Python/compare.py` is retired and always fails with directions to the standardized comparison command. Preserve historical artifacts; use audited standardized runs for new comparisons.

## Public workload profiles

`public-smoke` and `public-data` each contain 11 cases. Ten run the first five H2O-derived grouping queries on two key distributions. The last runs a full preprocessing workflow on all 1,599 rows of the pinned UCI red-wine dataset. Existing NIST profiles retain the 27 univariate checks. These additions do not extend NIST certification to other numerical APIs.

Prepare and run either profile with the same commands used above, replacing the profile name and output directory. Both profiles run offline after dependency installation; the small UCI source is bundled. Preparation verifies its original bytes and the derived CSV separately. No adapter downloads data during a run.

The H2O cases use an independent deterministic generator and smaller sizes. They are not an official H2O benchmark reproduction. See [public dataset provenance and workload boundaries](Fixtures/public-data.md) for query definitions, distributions and limitations. The wine case includes CSV reading, filtering, stable sorting, fitting population scaling and exporting the feature matrix, target vector and source-row IDs. Every measured result is checked against a separate Decimal reference, including target alignment. This measures descriptive preprocessing, not model training or predictive quality.
