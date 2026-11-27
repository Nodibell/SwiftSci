# SwiftSci Performance Benchmarks Index

> **Two-Tier Documentation Architecture**: Concise release-specific performance overviews live under [`PERFORMANCE/`](PERFORMANCE/) paired with [`ACCURACY/`](ACCURACY/). Raw cryptographic fingerprints, 6-decimal `Mean ± SD / CV% / IQR / z-score` dumps, and dataset byte tables live under [`Benchmarks/Reports/`](Benchmarks/Reports/). Use [`Benchmarks/README.md`](Benchmarks/README.md) to run and audit standardized suites.

The latest published version is **3.14.0**. See its [performance overview](PERFORMANCE/3.14.0.md), [deep performance audit](Benchmarks/Reports/v3.14.0/performance.deep.md), and [evidence index](Benchmarks/Reports/v3.14.0/README.md).

## Version Archive

| Version | Release Status | Scope | Overview | Deep Audit / Evidence | Companion |
|---|---|---|---|---|---|
| **`3.14.0`** | Latest (`v3.14.0`) | Six AC `production-default` campaigns (`10,300` samples across 4 engines) + `OneVsRestClassifier.predictProbability` (`2.94×` vs scikit-learn) + GBDT/RF `TreeSHAP` batch timings + `39/39` scientific suite + SwiftLLM KV-cache ablation | [Overview](PERFORMANCE/3.14.0.md) | [Deep Audit](Benchmarks/Reports/v3.14.0/performance.deep.md) · [Archive](Benchmarks/Reports/v3.14.0/README.md) | [Accuracy](ACCURACY/3.14.0.md) |
| **`3.13.3`** | Historical (`v3.13.3`, Superseded) | Six AC `production-default` campaigns (`10,300` samples across 4 engines) + 768-D MLP BLAS (`2.57×` fit) + `39/39` scientific suite + SwiftLLM KV-cache ablation | [Overview](PERFORMANCE/3.13.3.md) | [Deep Audit](Benchmarks/Reports/v3.13.3/performance.deep.md) · [Archive](Benchmarks/Reports/v3.13.3/README.md) | [Accuracy](ACCURACY/3.13.3.md) |
| **`3.13.2`** | Historical (`v3.13.2`) | Six audited standardized campaigns (`4,810` samples) + `39/39` scientific suite + SwiftLLM KV-cache ablation | [Overview](PERFORMANCE/3.13.2.md) | [Deep Audit](Benchmarks/Reports/v3.13.2/performance.deep.md) · [Archive](Benchmarks/Reports/v3.13.2/README.md) | [Accuracy](ACCURACY/3.13.2.md) |
| **`3.13.1`** | Historical (`v3.13.1`, Superseded) | Historical `standard`, `migration`, idle patched microbenchmarks (`2026-11-14`), OS-variance audit, and retired `--research` | [Overview](PERFORMANCE/3.13.1.md) | [Deep Audit](Benchmarks/Reports/v3.13.1/performance.deep.md) · [Index](Benchmarks/Reports/v3.13.1/README.md) | [Accuracy](ACCURACY/3.13.1.md) |
| **`3.13.0`** | Historical (`v3.13.0`) | Carries forward `3.12.0` audited baseline (`SwiftLLM` / `SwiftNLP` changes only; no independent classical run) | [Overview](PERFORMANCE/3.13.0.md) | [Methodology](Benchmarks/README.md) | [Accuracy](ACCURACY/3.13.0.md) |
| **`3.12.0`** | Historical (`v3.12.0`) | Audited `standard-3.12.0` & `migration-3.12.0` (`macOS 27.2`, Swift `6.4`) + legacy research tables | [Overview](PERFORMANCE/3.12.0.md) | [Methodology](Benchmarks/README.md) | [Accuracy](ACCURACY/3.12.0.md) |
| **`3.11.0`** | Historical (`v3.11.0`) | `standard-3.11.0` (`macOS 27.0.1`, Swift `6.4`) + 15-audit ML/Data `--research` baseline | [Overview](PERFORMANCE/3.11.0.md) | [Methodology](Benchmarks/README.md) | [Accuracy](ACCURACY/3.11.0.md) |
| **`3.10.3`** | Historical research (`v3.10.3`) | Methodology v2 initial `DenseColumn` / compensated precision / 1M DataFrame history | [Overview](PERFORMANCE/3.10.3.md) | [Methodology](Benchmarks/README.md) | [Accuracy](ACCURACY/3.10.3.md) |

See the [benchmark guide](Benchmarks/README.md) for reproduction.
