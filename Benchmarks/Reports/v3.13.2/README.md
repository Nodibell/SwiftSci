# SwiftSci 3.13.2 reproducibility archive

[Download the raw release evidence (GitHub Release asset)](https://github.com/Nodibell/SwiftSci/releases/download/v3.13.2/SwiftSci-v3.13.2-Stage-B-evidence.tar.gz). The archive preserves the verified run at commit `8039d02f9d63c27022a794dfff5bd43d2e8a9710`.

| Identifier | SHA-256 |
|---|---|
| Archive | `8a7bdc86bd3da7e5a92d1eb68b98ab283f7f9c3b680cc244dabc1a6ac0458eea` |
| Source tree | `953e773629eabf731530b82edd6d74813e8d5f4b22bfec25b412ba9951eb4a9c` |
| Worker | `535b15de13233d243d3f76ef0072cf1d309ce47ab5db5ae0b61894d001abd734` |
| Scientific JSON | `9c3f79b7ac7afdd833de6a130237ff60a035946dc3807b1da2e397d7d7f241eb` |

The archive contains source and worker fingerprints, six resolved benchmark plans and certificates, CPU acceptance and regression decisions, scientific validation, control results and test logs. Some raw filenames and fields retain internal run labels. The commit and fingerprints identify the approved release.

## Deep Audit Reports

- [3.13.2 Deep Accuracy & Conformance Audit](accuracy.deep.md) — cryptographic fingerprints, all 27 NIST StRD dataset errors, G-037 residual identity verification, rolling-origin backtests, all 39 raw scientific workload gate IDs, pandas Wampler comparator mismatches, skipped conditional Llama checks, and G-035/G-036 deferred scope.
- [3.13.2 Deep Performance Audit](performance.deep.md) — cryptographic fingerprints, six standardized campaign summary tables (`4,810` validated samples), full 17-dataset byte manifest, 6-decimal `median ± SD (ms)` tables for `standard` and `standard-production-default`, all 33 scientific workload `median (IQR)` stages, and SwiftLLM KV-cache ablation timings.

[Runner guide](../../README.md) · [Performance](../../../PERFORMANCE/3.13.2.md) · [Accuracy](../../../ACCURACY/3.13.2.md)
