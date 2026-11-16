# SwiftSci Scientific Accuracy & Conformance Index

> **Versioned Documentation Architecture**: Full release-specific numerical validation matrices, conformance certificates, and methodology records live under [`ACCURACY/`](ACCURACY/) paired with [`PERFORMANCE/`](PERFORMANCE/). See [`Benchmarks/README.md`](Benchmarks/README.md) and [`Benchmarks/Specs/acceptance.json`](Benchmarks/Specs/acceptance.json) for acceptance gates.

The latest published version is **3.13.2**. See its [release report](ACCURACY/3.13.2.md) and the [evidence index](Benchmarks/Reports/v3.13.2/README.md).

## Version archive

| Version | Release status | Scope | Report | Companion |
|---|---|---|---|---|
| **3.13.2** | Latest published | NIST, scientific workloads, forecasting and CPU acceptance | [Report](ACCURACY/3.13.2.md) | [Companion](PERFORMANCE/3.13.2.md) |
| **`3.13.1`** | Published (`v3.13.1`, Superseded) | `certification` (`27/27`), `dataframe-conformance` (`24/24`), `scientific-workflow` (`4/4`), CSV null-token round-trip, historical snapshot before G-033/G-037 fixes | [`ACCURACY/3.13.1.md`](ACCURACY/3.13.1.md) | [`PERFORMANCE/3.13.1.md`](PERFORMANCE/3.13.1.md) |
| **`3.13.0`** | Released (`v3.13.0`) | Carries forward `3.12.0` classical baseline + packed 4-bit/8-bit LLM, parser bounds, thread-safe `AppleSentenceEmbeddingEngine` | [`ACCURACY/3.13.0.md`](ACCURACY/3.13.0.md) | [`PERFORMANCE/3.13.0.md`](PERFORMANCE/3.13.0.md) |
| **`3.12.0`** | Released (`v3.12.0`) | Audited `standard-3.12.0` / `migration-3.12.0`, deterministic FNV-1a text embeddings, GPU BCE gradient, wire-protocol hardening | [`ACCURACY/3.12.0.md`](ACCURACY/3.12.0.md) | [`PERFORMANCE/3.12.0.md`](PERFORMANCE/3.12.0.md) |
| **`3.11.0`** | Released (`v3.11.0`) | Audits 1–15 mathematical remediations (TreeSHAP/KernelSHAP/LIME, ETS/Kalman, DecisionTree/LinearRegression, TargetEncoder, WordNet/VADER) | [`ACCURACY/3.11.0.md`](ACCURACY/3.11.0.md) | [`PERFORMANCE/3.11.0.md`](PERFORMANCE/3.11.0.md) |
| **`3.10.3`** | Released (`v3.10.3`) | Blocked centered ANOVA, Kahan-Babuška dot products, PCA total-variance EVR, Apache Parquet spec conformance | [`ACCURACY/3.10.3.md`](ACCURACY/3.10.3.md) | [`PERFORMANCE/3.10.3.md`](PERFORMANCE/3.10.3.md) |

See the [benchmark guide](Benchmarks/README.md) for reproduction.
