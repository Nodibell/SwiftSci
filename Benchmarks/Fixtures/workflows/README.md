# Public scientific and fitted-model workflows

These original, bounded fixtures check complete public-API paths on Apple silicon. They establish no predictive-quality or production-performance baseline. Every output element is checked against an independent reference. The [scientific profile](../../Specs/profiles/scientific-workflow.json) contains four cases. The [persistence profile](../../Specs/profiles/persisted-workflow.json) contains six cases.

## Scientific observations to a numerical result

The worker stages two CSV files, reads them with `DataFrame(csv:options:)`, filters observations with quality at least one, and performs an inner join by site. CSV type overrides declare IDs and sites as Int64 and measurements and quality as Double. This avoids treating a zero/one quality field as an inferred Boolean. Missing quality values are excluded. Duplicate calibration keys multiply matching observations. Unmatched observations disappear. The worker sorts by observation ID and then calibration ID, exports both nested and flat matrices in the declared feature order, and checks that those representations agree. It then fits `LinearRegression(device: .cpu)` and predicts through the public API.

One case uses unique calibration keys as a control. The other cases include missing quality, rejected observations, an unmatched site, duplicate join keys, reversed feature order, and shuffled source rows. Output contains source row count, filtered count, joined count, width, all observation IDs, all calibration IDs, the row-major matrix, aligned targets, intercept, ordered coefficients, predictions, signed residuals, and residual sum of squares.

The [independent reference](../../Tools/workflow_reference.py) parses CSV through Python's standard library, joins with a nested scalar loop, and solves the small full-rank least-squares problem with exact rational arithmetic. It does not import pandas, NumPy, SciPy, or SwiftSci. It rejects rank-deficient inputs rather than manufacturing coefficients. The comparison worker uses pandas and SciPy least squares. The canonical sort is part of the contract, so this workload makes no claim about a join's unspecified output ordering.

## Training-only preprocessing and persistence

Six training observations fit a public `RegressionPipeline` containing `StandardScaler` and CPU `LinearRegression`. Three disjoint held-out observations supply predictions and residuals. The scaler uses population variance. The fixture has distinct feature and row identities. No identical feature row crosses partitions.

Each of three routes has a paired case with changed held-out features and targets. Fitted means, scales, coefficients, and training predictions must remain unchanged. Every case compares those values to independently calculated answers. Predictions must not mutate fitted state.

| Route | Public APIs exercised | Claim and limit |
| --- | --- | --- |
| `fit` | Pipeline fit, scaler transform, and pipeline predict | Qualifies the fitted public path independently of export failures. |
| `native` | Regressor `save(to:)`, fresh-process `LinearRegression.load(from:device:)`, and predict | Qualifies native regressor persistence using already transformed held-out inputs. It does not claim native scaler or pipeline persistence. |
| `coreml` | Fitted scaler export, fitted regressor `exportCoreML`, composite pipeline export, Core ML compile/load/predict | Attempts complete persisted preprocessing and prediction from raw held-out inputs in a new CPU-only process. Export or runtime failures fail the case. |

SwiftSci currently has no native save/load API for the fitted preprocessing pipeline. The worker does not add a replacement implementation. The Core ML route composes its existing public export APIs, preserving the fitted scaler and model together. The native route remains separate so a successful export cannot hide a native-loader defect.

Output order is training count, held-out count, width, training IDs, held-out IDs, fitted means, fitted scales, transformed training matrix, transformed held-out matrix, intercept, ordered coefficients, training predictions, held-out predictions, and held-out residuals. Reload routes append all reloaded predictions. The final value is one only after fitted-state checks pass. Reload routes additionally require a successful child with a distinct process ID, matching model checksum, and complete finite predictions.

The reference solves OLS in raw coordinates using exact rational arithmetic. Ninety-digit Decimal arithmetic converts its coefficients to the standardized coordinate system and computes population scales. No answers appear in worker input files. Expected tolerance is `1e-8 + 1e-10 * abs(expected)` for these small, full-rank Double cases. Counts and IDs are small integers, so an integer mistake cannot fall within that tolerance. This bound does not apply to ill-conditioned systems or arbitrary models.

The NumPy/SciPy comparison fits population scaling and least squares, persists mean, scale, coefficients, and feature names in NPZ, then loads and predicts in a new Python process. It supplies a numerical and process-isolation comparison. It does not exercise SwiftSci JSON or Core ML formats. Timing comparisons between these deployment routes would be misleading.

## Evidence and failure handling

Each worker retains a directory beside its response for each sample. Scientific samples retain staged CSV files. Completed operations retain `actual.json`, including when comparison against the reference subsequently fails. Model samples retain `before-reload.json`. Reload samples also retain the model, checksum-bound child request, child response, and log when available. A failing export or child process preserves these diagnostics and returns a failed benchmark response. The runner retains the failure and continues other cases.

The certificate binds normal requests, references, worker responses, and source identity through the existing protocol. Supplemental sample files aid diagnosis; the current certificate audit does not independently hash every file in those directories. The worker verifies the model checksum before loading and validates reloaded predictions against the independent reference. Do not describe the supplemental directory as a separately certified archive.

All timing includes execution through complete output extraction. Scientific timing includes staging writes and CSV reads. Persistence timing includes training, serialization, child startup, compilation when applicable, and prediction. These are diagnostic timings. Parent-process peak RSS excludes the child's peak and does not measure complete deployment memory. Each child has a 60-second deadline within the profile's outer timeout.

Both profiles join the CPU acceptance tier and retain known failures. An unsupported or broken path is never counted as a pass. Production fixes belong on the separate repair branch.

## Regenerate and run

Run these commands from the repository root. The original fixtures use the repository's MIT license. [Source checksums](sources.lock.json) bind the generator, validator, and independent reference. Regeneration is deterministic.

```sh
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/build_workflow_fixtures.py
Benchmarks/.venv-standardized/bin/python -m unittest discover -s Benchmarks/Tests -p test_public_workflows.py
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py prepare --profile scientific-workflow
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py prepare --profile persisted-workflow
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py build
```

Run each profile with a new output directory and the recorded worker from the build. Replace the profile and output directory for the second run.

```sh
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py run \
  --profile scientific-workflow --engines swiftsci,pandas \
  --swift-worker "$HOME/Library/Caches/SwiftSci/standardized-benchmarks/derived/Build/Products/Release/SwiftSciBenchmarkWorker" \
  --python Benchmarks/.venv-standardized/bin/python \
  --output Benchmarks/Runs/scientific-workflow-unique-name
```

Controller tests cover independent answers, deterministic replay, join multiplicity, feature order, held-out perturbations, malformed inputs, provenance, per-element corruption, fresh processes, and rejection of corrupt persisted comparator files. Actual Swift execution remains necessary to qualify the public APIs.

The CI workflow also runs [actual-worker sensitivity checks](../../Tests/check_public_workflow_workers.py) after acceptance, even when another profile fails. Each eligible case must pass unchanged before its worker receives wrong answers, truncated answers, malformed input, and an embedded-answer field. A baseline failure remains ineligible for this sensitivity claim. It is not counted as successful corruption rejection.
