# SwiftSci Performance Benchmarks Index

> **Versioned Documentation Architecture**: Full release-specific benchmark tables, methodology records, and raw environment metadata live under [`PERFORMANCE/`](PERFORMANCE/) paired with [`ACCURACY/`](ACCURACY/). Use [`Benchmarks/README.md`](Benchmarks/README.md) to run and audit standardized suites.

> [!NOTE]
> **Latest Candidate Status (`v3.13.2-candidate`) & Historical Provenance (`3.13.1`, `3.13.0`)**:
> - [`PERFORMANCE/3.13.2-candidate.md`](PERFORMANCE/3.13.2-candidate.md) documents the **Stage A AC-power candidate** (base commit `0cf99f0470d0757d0a02bd3244c62d821ab144b0` `v3.13.1`, dirty tree `5abd55ba...`, Apple M4 Pro `macOS 27.2.1 / 26B131`, Swift `6.4.2`) after retiring obsolete `--research` harnesses and adding scientific validation gates. Final Stage B freeze requires commit/tag approval and a clean-tree re-run.
> - [`PERFORMANCE/3.13.1.md`](PERFORMANCE/3.13.1.md) preserves the superseded `3.13.1` report (`2026-11-14`).
> - **`v3.13.0` has no independent classical/standardized benchmark run** on disk (`Benchmarks/Runs/` contains `3.11.0`, `3.12.0`, `2026-11-14`, and `v3.13.2-candidate` runs). [`PERFORMANCE/3.13.0.md`](PERFORMANCE/3.13.0.md) carries forward the audited `v3.12.0` baseline alongside `v3.13.0`'s `SwiftLLM` / `SwiftNLP` changes.

---

## Version Archive

| Version | Release Status | Benchmark Methodology & Profile Scope | Report | Accuracy Companion |
| :--- | :---: | :--- | :--- | :--- |
| **`3.13.2-candidate`** | Stage A Candidate (`base 0cf99f0`) | Audited `standard`, `standard-production-default`, `nist`, `public-data`, `migration`, `tabular-migration`, `acceptance-cpu`, `scientific_validation_suite.py` (`39/39`) on AC power; legacy `--research` retired | [**`PERFORMANCE/3.13.2-candidate.md`**](PERFORMANCE/3.13.2-candidate.md) | [`ACCURACY/3.13.2-candidate.md`](ACCURACY/3.13.2-candidate.md) |
| **`3.13.1`** | Published (`v3.13.1`, Superseded) | Historical `standard`, `migration`, `certification`, `dataframe-conformance`, `scientific-workflow`, idle patched microbenchmarks (`2026-11-14`), and retired `--research` | [`PERFORMANCE/3.13.1.md`](PERFORMANCE/3.13.1.md) | [`ACCURACY/3.13.1.md`](ACCURACY/3.13.1.md) |
| **`3.13.0`** | Released (`v3.13.0`) | Carries forward `3.12.0` audited baseline (`SwiftLLM` / `SwiftNLP` changes only; no independent classical run) | [`PERFORMANCE/3.13.0.md`](PERFORMANCE/3.13.0.md) | [`ACCURACY/3.13.0.md`](ACCURACY/3.13.0.md) |
| **`3.12.0`** | Released (`v3.12.0`) | Audited `standard-3.12.0` & `migration-3.12.0` (`macOS 27.2`, Swift `6.4`) + legacy research tables | [`PERFORMANCE/3.12.0.md`](PERFORMANCE/3.12.0.md) | [`ACCURACY/3.12.0.md`](ACCURACY/3.12.0.md) |
| **`3.11.0`** | Released (`v3.11.0`) | `standard-3.11.0` (`macOS 27.0.1`, Swift `6.4`) + 15-audit ML/Data `--research` baseline | [`PERFORMANCE/3.11.0.md`](PERFORMANCE/3.11.0.md) | [`ACCURACY/3.11.0.md`](ACCURACY/3.11.0.md) |
| **`3.10.3`** | Released (`v3.10.3`) | Methodology v2 initial `DenseColumn` / compensated precision / 1M DataFrame history | [`PERFORMANCE/3.10.3.md`](PERFORMANCE/3.10.3.md) | [`ACCURACY/3.10.3.md`](ACCURACY/3.10.3.md) |

---

The historical [v3.13.1 performance report](PERFORMANCE/3.13.1.md) preserves its timings, profiles, and environment caveats.

## Reproduction

Use the [benchmark guide](Benchmarks/README.md) for standardized runs and strict same-environment comparisons. The [Stage B protocol](Benchmarks/Reports/v3.13.2-candidate/FINAL_BENCHMARK_COMMANDS.md) lists the full release sweep.
