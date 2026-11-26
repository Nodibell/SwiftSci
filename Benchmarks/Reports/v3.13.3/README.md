# SwiftSci 3.13.3 Reproducibility Archive & Evidence Index

This directory preserves the verified release-gate and AC-powered `production-default` comparison runs for **SwiftSci `3.13.3`** on branch `release/3.13.3` (base commit `dd3a8d980c81bb2e6f15ce4fb4400bd1bb959320`, `dirty: true` with `3.13.3` source and test updates).

## Cryptographic Fingerprints (SHA-256)

| Artifact / Identifier | Path / Source | SHA-256 |
|---|---|---|
| Source tree | `summary.json` / `suite.json` (`source.tree_sha256`) | `334c6d57af4ceec2d8f4c1ca8e54f9d3f62bab2e817b56d4d791294639171b04` |
| Benchmark worker (on disk) | `SwiftSciBenchmarkWorker` (Release) | `a6c568e291bc9358fd1b7633347b71840858be7245ff0548dd9ebe2af1221776` |
| Benchmark worker (gate JSON record) | `scientific_validation.json` / `pretrained-inspection/controls.json` | `c907ae3c5f6b8b82d0de7e6047391d8bc109084bc4a06179ec3a51a3320461a2` |
| Release-gate summary | `Benchmarks/Runs/v3.13.3-release-gate/summary.json` | `8aa5845f8e97dc572489833652ff9d767d5c6d2ed788cbae2eb0b3b65b197a3a` |
| Scientific validation JSON | `Benchmarks/Runs/v3.13.3-release-gate/scientific_validation.json` | `0184a50235c818f9f5daa5e516e6d83891210644b6d259251f706e9011cff62d` |
| Release-gate acceptance JSON | `Benchmarks/Runs/v3.13.3-release-gate/acceptance/acceptance.json` | `b59ee8a1d2aa81e46e77cb125df20402034c240f29b0c76c4c2b4e90b8ce8112` |
| Comparison suite manifest | `Benchmarks/Runs/v3.13.3-comparison/suite.json` | `dc94c9a2757a3b41255a2c97d3e025dfe028c8e35aa7763f686ba753f38a4643` |
| Comparison measurements JSON | `Benchmarks/Runs/v3.13.3-comparison/measurements.json` | `f1d57e17c99916044094f277f6bca40f4d6b1cd905fbdf9d25746fb6b6847237` |
| Comparison CPU acceptance JSON | `Benchmarks/Runs/v3.13.3-comparison/cpu-acceptance/acceptance.json` | `043a6f6ff59f26e7ab65435d27e7e7f5648234f1359834bd175bef4b21360035` |

## Run Provenance Notes

- **AC-Powered Standardized Comparison (`v3.13.3-comparison`)**: Executed all `6/6` profiles (`standard`, `extended`, `public-data`, `nist`, `migration`, `tabular-migration`) under `production-default` (`5` rotating fresh worker processes per engine-case, `25` samples per engine-case, `10,300` total validated samples, `0` failures) with `power_start = AC`, `power_end = AC`, and `power_unchanged = true` across every profile boundary, plus `cpu-acceptance` (`16` profiles, `202/202` SwiftSci cases passed).
- **Release-Gate Suite (`v3.13.3-release-gate`)**: `build`, `inventory`, `unittest`, `acceptance-cpu` (`14/16` strict profiles; `202/202` SwiftSci cases), `pretrained-controls` (`60/60`), `workflow-controls` (`100/100`), and `scientific-validation` (`39/39` workloads, AC power, `legacy-capped`) all passed. The trailing `comparison-suite` step inside `release-gate` failed on battery power (`power_state.source = "battery"`, `exit_code = 1`) and was superseded by the clean AC-powered `v3.13.3-comparison` run above.
- **Wave 2 Verification & MLP Bench**: `xcodebuild test` + `swift test -c release` (`1,874` test cases, `0` failures, `7` env-gated skips), `verify_doc_coverage.py` (`100.00%`, `2,551/2,551` matched declarations documented), `CookbookVerification` (`71/71` recipes passed), and `MLPClassifier` 768-D benchmark (`40.312 ms` vs scikit-learn `103.639 ms`, **2.57×** faster fit).

## Deep Audit Reports

- [3.13.3 Deep Accuracy & Conformance Audit](accuracy.deep.md) — cryptographic fingerprints, 16-profile acceptance summary, all 27 NIST StRD dataset errors, numerical gradient & extreme logit stability checks, rolling-origin backtests, all 39 raw scientific workload gate IDs, pandas Wampler comparator mismatches, and G-035/G-036 deferred scope.
- [3.13.3 Deep Performance Audit](performance.deep.md) — cryptographic fingerprints, all 6 AC-powered `production-default` comparison campaign tables (`10,300` validated samples across `412` engine-case rows), 17-dataset byte manifest, all 33 scientific workload `median (IQR)` stages, SwiftLLM KV-cache ablation timings, and raw `MLPBench` 768-D samples.

[Runner guide](../../README.md) · [Performance](../../../PERFORMANCE/3.13.3.md) · [Accuracy](../../../ACCURACY/3.13.3.md)
