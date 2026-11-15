# SwiftSci Scientific Accuracy & Conformance Index

> **Versioned Documentation Architecture**: Full release-specific numerical validation matrices, conformance certificates, and methodology records live under [`ACCURACY/`](ACCURACY/) paired with [`PERFORMANCE/`](PERFORMANCE/). See [`Benchmarks/README.md`](Benchmarks/README.md) and [`Benchmarks/Specs/acceptance.json`](Benchmarks/Specs/acceptance.json) for acceptance gates.

> [!NOTE]
> **Latest Candidate Status (`v3.13.2-candidate`) & Historical Provenance (`3.13.1`, `3.13.0`)**:
> - [`ACCURACY/3.13.2-candidate.md`](ACCURACY/3.13.2-candidate.md) documents the **Stage A AC-power correctness & scientific validation candidate** (`base 0cf99f0470...` `v3.13.1`, dirty tree): **G-037 FIXED AND VERIFIED** (`ARIMA`/`SARIMA` closed-loop original-scale `fittedValues`), **G-033** MA invertibility & finite validation, `SwiftStats` `ddof=1` DocC alignment, **G-035/G-036** `TreeSHAP` DocC/deferrals, `39/39` scientific gates, and `14/16` CPU acceptance profiles passing (`2 profiles contain 4 known `pandas` Wampler 4/5 comparator failures).
> - [`ACCURACY/3.13.1.md`](ACCURACY/3.13.1.md) preserves the superseded `v3.13.1` snapshot with warning banner.
> - **`v3.13.0` has no independent classical/standardized benchmark run** on disk. [`ACCURACY/3.13.0.md`](ACCURACY/3.13.0.md) retains the audited `v3.12.0` classical baseline alongside `v3.13.0`'s packed LLM, binary-parser bounds, and embedding safety checks.

---

## Version Archive

| Version | Release Status | Validation & Conformance Scope | Report | Performance Companion |
| :--- | :---: | :--- | :--- | :--- |
| **`3.13.2-candidate`** | Stage A Candidate (`base 0cf99f0`) | **G-037 FIXED AND VERIFIED**, G-033, SwiftStats DocC, G-035/G-036 scope, `39/39` `scientific_validation_suite.py` + rolling-origin backtests, `14/16` CPU acceptance (`0` SwiftSci failures), `60` pretrained + `100` control checks | [**`ACCURACY/3.13.2-candidate.md`**](ACCURACY/3.13.2-candidate.md) | [`PERFORMANCE/3.13.2-candidate.md`](PERFORMANCE/3.13.2-candidate.md) |
| **`3.13.1`** | Published (`v3.13.1`, Superseded) | `certification` (`27/27`), `dataframe-conformance` (`24/24`), `scientific-workflow` (`4/4`), CSV null-token round-trip, historical snapshot before G-033/G-037 fixes | [`ACCURACY/3.13.1.md`](ACCURACY/3.13.1.md) | [`PERFORMANCE/3.13.1.md`](PERFORMANCE/3.13.1.md) |
| **`3.13.0`** | Released (`v3.13.0`) | Carries forward `3.12.0` classical baseline + packed 4-bit/8-bit LLM, parser bounds, thread-safe `AppleSentenceEmbeddingEngine` | [`ACCURACY/3.13.0.md`](ACCURACY/3.13.0.md) | [`PERFORMANCE/3.13.0.md`](PERFORMANCE/3.13.0.md) |
| **`3.12.0`** | Released (`v3.12.0`) | Audited `standard-3.12.0` / `migration-3.12.0`, deterministic FNV-1a text embeddings, GPU BCE gradient, wire-protocol hardening | [`ACCURACY/3.12.0.md`](ACCURACY/3.12.0.md) | [`PERFORMANCE/3.12.0.md`](PERFORMANCE/3.12.0.md) |
| **`3.11.0`** | Released (`v3.11.0`) | Audits 1–15 mathematical remediations (TreeSHAP/KernelSHAP/LIME, ETS/Kalman, DecisionTree/LinearRegression, TargetEncoder, WordNet/VADER) | [`ACCURACY/3.11.0.md`](ACCURACY/3.11.0.md) | [`PERFORMANCE/3.11.0.md`](PERFORMANCE/3.11.0.md) |
| **`3.10.3`** | Released (`v3.10.3`) | Blocked centered ANOVA, Kahan-Babuška dot products, PCA total-variance EVR, Apache Parquet spec conformance | [`ACCURACY/3.10.3.md`](ACCURACY/3.10.3.md) | [`PERFORMANCE/3.10.3.md`](PERFORMANCE/3.10.3.md) |

---

The historical [v3.13.1 accuracy report](ACCURACY/3.13.1.md) preserves that release’s conformance results, CSV round-trip checks, and retired research outputs.

## Reproduction

Use the [benchmark guide](Benchmarks/README.md) to prepare fixtures and run conformance checks. The [Stage B protocol](Benchmarks/Reports/v3.13.2-candidate/FINAL_BENCHMARK_COMMANDS.md) includes the clean-tree scientific validation and CPU acceptance commands.
