# SwiftSci 3.14.0 Reproducibility Archive & Evidence Index

This directory preserves the verified release-gate and AC-powered `production-default` comparison runs for **SwiftSci `3.14.0`** on branch `release/3.14.0` (base commit `10c11b0c586663484e0f982f7afd52834d3924d8`, `dirty: true` with `3.14.0` source and test updates).

## Cryptographic Fingerprints (SHA-256)

| Artifact / Identifier | Path / Source | SHA-256 |
|---|---|---|
| Source tree | `summary.json` / `suite.json` (`source.tree_sha256`) | `c6cd7882049ccb65c485c5d113880a44615a5fbb553137ef13abe7bfb79fa2b7` |
| Benchmark worker (Release) | `SwiftSciBenchmarkWorker` (on disk & `scientific_validation.json`) | `0f75343624dbaf306c452262adb5e675b2436cf60aa538e505b336edd74e9968` |
| Release-gate summary | `Benchmarks/Runs/v3.14.0-release-gate/summary.json` | `7e771fa8d48b96a3b942bf7ac195f5c2ac94d3beafbd0ed5913e759d8273aefa` |
| Scientific validation JSON | `Benchmarks/Runs/v3.14.0-release-gate/scientific_validation.json` | `e0d38b0d9aa527d36ca6b84bde511278014e0f7ca8bdd8b5c7a8f9e8cb22dbdc` |
| Release-gate acceptance JSON | `Benchmarks/Runs/v3.14.0-release-gate/acceptance/acceptance.json` | `d09e640e37638601761cf36829e36d193232011bb28551c19f2828f5dc2df15b` |
| Comparison suite manifest | `Benchmarks/Runs/v3.14.0-release-gate/comparison/suite.json` | `ab111e4068d067e21931894e0c7465b8d0575f21a1b3d7cbfe67a543af215222` |
| Comparison measurements JSON | `Benchmarks/Runs/v3.14.0-release-gate/comparison/measurements.json` | `751a7d3c63905827db358c560ddcbaade8462a80a076cd78177cf75e6e5a0eea` |
| Comparison CPU acceptance JSON | `Benchmarks/Runs/v3.14.0-release-gate/comparison/cpu-acceptance/acceptance.json` | `f28fb2282ed026b6804a4011e907d2fb42852cf6f88467f89330858ebb21d5da` |
| Pretrained inspection controls | `Benchmarks/Runs/v3.14.0-release-gate/pretrained-inspection/controls.json` | `b45fb4a526421ca31a704e05c96ff62d977957158d45900b86ef70ace65be58b` |
| Workflow mutation controls | `Benchmarks/Runs/v3.14.0-release-gate/workflow-controls/controls.json` | `5182daeefe3dccd82daef818a3901c68e01083eafdcba637c525873147231e5f` |
| `v3140_bench` Release binary | `v3140_bench/.build/release/V3140Bench` | `5c8248e16f8de95e6273d494038d22b3d043194b9c564e86b8447eba12ce5806` |
| `v3140_bench` data fixture | `v3140_bench/bench_data.json` | `76fc5dab4f6456e739c6e2e74e503f072d8630bcfb49701ca3ab33c6179eb349` |
| `v3140_bench` Python script | `v3140_bench/bench_python.py` | `5736159044422b787caee3726aaba98b638f19336cde829a77787e9e29d1c91b` |
| `v3140_bench` Swift source | `v3140_bench/Sources/V3140Bench/main.swift` | `8f2f700a0aaeb82539d51494cabdb6e3e47c87701313a1fc82378d6d5f23bbbe` |
| `treeshap_oracle` Release binary | `treeshap_oracle/.build/release/TreeSHAPOracle` | `3fd1dd98c7e41bc29068ead30dcd20dbf66c58dda6ab42c6ab2be0f5cc1c5f23` |
| `treeshap_oracle` payload | `treeshap_oracle/oracle_payload.json` | `544a1448731f1d6655ec2b04a996749f1d551ea2c454e8299a83da1c30c96f04` |

## Run Provenance Notes

- **Release-Gate Suite (`v3.14.0-release-gate`, `all_passed: true`)**: All 8 steps (`build`, `inventory`, `unittest`, `acceptance-cpu`, `pretrained-controls`, `workflow-controls`, `scientific-validation`, `comparison-suite`) passed cleanly on AC power (`power_state.source = "AC"`).
- **AC-Powered Standardized Comparison (`v3.14.0-release-gate/comparison`)**: Executed all `6/6` profiles (`standard`, `extended`, `public-data`, `nist`, `migration`, `tabular-migration`) under `production-default` (`5` rotating fresh worker processes per engine-case, `25` samples per engine-case, `10,300` total validated samples across `412` engine-case rows, `0` failures) with `power_start = AC`, `power_end = AC`, and `power_unchanged = true` across every profile boundary, plus `cpu-acceptance` (`16` profiles, `202/202` SwiftSci cases passed).
- **Auxiliary & Scientific Gates**: `scientific-validation` passed `39/39` workloads, `3/3` rolling-origin backtests, and `2/2` expected-error checks (`all_gates_passed: true`, AC power, `legacy-capped`). `pretrained-controls` passed `60/60`, and `workflow-controls` passed `100/100`.
- **Wave 3 Verification, TreeSHAP Oracle & `v3140_bench`**: `xcodebuild test` + `swift test -c release` (`1,919` total: `1,912` passed / `2,814` dynamic runs, `7` env-gated skips, `0` failures), `verify_doc_coverage.py` (`100.00%`, `2,641/2,641` matched declarations documented) + `15/15` DocC archives (`--warnings-as-errors`, `0` placeholders), `CookbookVerification` (`71/71` recipes passed), Python `shap.TreeExplainer` (`shap 0.54.1`) 5-model oracle cross-check (`<= 1.776e-15`), and `v3140_bench` Release timings (`OneVsRestClassifier.predictProbability` `0.069 ms` vs scikit-learn `0.203 ms`, **2.94×** faster).

## Deep Audit Reports

- [3.14.0 Deep Accuracy & Conformance Audit](accuracy.deep.md) — cryptographic fingerprints, 16-profile acceptance summary, all 27 NIST StRD dataset errors, `treeshap_oracle` (`shap 0.54.1`) 5-model parity table, `G-035`–`G-040` & Core ML Vision parity gates, rolling-origin backtests, all 39 raw scientific workload gate IDs, and pandas Wampler comparator mismatches.
- [3.14.0 Deep Performance Audit](performance.deep.md) — cryptographic fingerprints, all 6 AC-powered `production-default` comparison campaign tables (`10,300` validated samples across `412` engine-case rows), 17-dataset byte manifest, all 33 scientific workload `median (IQR)` stages, SwiftLLM KV-cache ablation timings, and `v3140_bench` Release timings.

[Runner guide](../../README.md) · [Performance](../../../PERFORMANCE/3.14.0.md) · [Accuracy](../../../ACCURACY/3.14.0.md)
