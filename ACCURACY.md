# SwiftSci Scientific Accuracy & Conformance Index

> **Two-Tier Documentation Architecture**: Concise release-specific accuracy overviews live under [`ACCURACY/`](ACCURACY/) paired with [`PERFORMANCE/`](PERFORMANCE/). Raw cryptographic fingerprints, per-workload gate IDs (`ts_*`, `ml_*`, etc.), and deep conformance records live under [`Benchmarks/Reports/`](Benchmarks/Reports/). See [`Benchmarks/README.md`](Benchmarks/README.md) and [`Benchmarks/Specs/acceptance.json`](Benchmarks/Specs/acceptance.json) for acceptance gates.

The latest published version is **3.14.0**. See its [accuracy overview](ACCURACY/3.14.0.md), [deep accuracy audit](Benchmarks/Reports/v3.14.0/accuracy.deep.md), and [evidence index](Benchmarks/Reports/v3.14.0/README.md).

## Version Archive

| Version | Release Status | Scope | Overview | Deep Audit / Evidence | Companion |
|---|---|---|---|---|---|
| **`3.14.0`** | Latest (`v3.14.0`) | NIST (`27/27`), `39/39` scientific workloads, G-035–G-040 (`TreeSHAP` GBDT & per-class probability `<= 1.78e-15` vs `shap.TreeExplainer`, `ReActAgent` chat templates + `GenerationConfiguration`, `SwiftVision` dataset/crop loaders, `TrainedVisionClassifier` + `OneVsRestClassifier` + Core ML `VisionFeaturePrint.Scene`), `6/6` AC `production-default` certs (`10,300` samples), `202/202` CPU acceptance | [Overview](ACCURACY/3.14.0.md) | [Deep Audit](Benchmarks/Reports/v3.14.0/accuracy.deep.md) · [Archive](Benchmarks/Reports/v3.14.0/README.md) | [Performance](PERFORMANCE/3.14.0.md) |
| **`3.13.3`** | Historical (`v3.13.3`, Superseded) | NIST (`27/27`), `39/39` scientific workloads, G-042–G-053 & G-038 fixes (MLP BLAS/gradients, HistGBDT multiclass, LinearSVC, Platt/Calibrated, Vision, Gemma 4, ReAct), `6/6` AC `production-default` certs (`10,300` samples), `202/202` CPU acceptance | [Overview](ACCURACY/3.13.3.md) | [Deep Audit](Benchmarks/Reports/v3.13.3/accuracy.deep.md) · [Archive](Benchmarks/Reports/v3.13.3/README.md) | [Performance](PERFORMANCE/3.13.3.md) |
| **`3.13.2`** | Historical (`v3.13.2`) | NIST (`27/27`), `39/39` scientific workloads, G-033/G-037 fixes, `6/6` standardized certs (`4,810` samples), `202/202` CPU acceptance | [Overview](ACCURACY/3.13.2.md) | [Deep Audit](Benchmarks/Reports/v3.13.2/accuracy.deep.md) · [Archive](Benchmarks/Reports/v3.13.2/README.md) | [Performance](PERFORMANCE/3.13.2.md) |
| **`3.13.1`** | Historical (`v3.13.1`, Superseded) | `certification` (`27/27`), `dataframe-conformance` (`24/24`), `scientific-workflow` (`4/4`), CSV null-token round-trip, pre-G-033/G-037 snapshot | [Overview](ACCURACY/3.13.1.md) | [Deep Audit](Benchmarks/Reports/v3.13.1/accuracy.deep.md) · [Index](Benchmarks/Reports/v3.13.1/README.md) | [Performance](PERFORMANCE/3.13.1.md) |
| **`3.13.0`** | Historical (`v3.13.0`) | Carries forward `3.12.0` classical baseline + packed 4-bit/8-bit LLM, parser bounds, serialized `AppleSentenceEmbeddingEngine` | [Overview](ACCURACY/3.13.0.md) | [Methodology](Benchmarks/README.md) | [Performance](PERFORMANCE/3.13.0.md) |
| **`3.12.0`** | Historical (`v3.12.0`) | Audited `standard-3.12.0` / `migration-3.12.0`, deterministic FNV-1a text embeddings, GPU BCE gradient, wire-protocol hardening | [Overview](ACCURACY/3.12.0.md) | [Methodology](Benchmarks/README.md) | [Performance](PERFORMANCE/3.12.0.md) |
| **`3.11.0`** | Historical (`v3.11.0`) | Audits 1–15 mathematical remediations (TreeSHAP/KernelSHAP/LIME, ETS/Kalman, DecisionTree/LinearRegression, TargetEncoder, WordNet/VADER) | [Overview](ACCURACY/3.11.0.md) | [Methodology](Benchmarks/README.md) | [Performance](PERFORMANCE/3.11.0.md) |
| **`3.10.3`** | Historical research (`v3.10.3`) | Blocked centered ANOVA, Kahan-Babuška dot products, PCA total-variance EVR, Apache Parquet spec conformance | [Overview](ACCURACY/3.10.3.md) | [Methodology](Benchmarks/README.md) | [Performance](PERFORMANCE/3.10.3.md) |

See the [benchmark guide](Benchmarks/README.md) for reproduction.
