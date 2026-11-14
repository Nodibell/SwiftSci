# SwiftSci Scientific Accuracy & Conformance Index

> **Versioned Documentation Architecture**: Full release-specific numerical validation matrices, conformance certificates, and methodology records live under [`ACCURACY/`](ACCURACY/) paired with [`PERFORMANCE/`](PERFORMANCE/). See [`Benchmarks/README.md`](Benchmarks/README.md) and [`Benchmarks/Specs/acceptance.json`](Benchmarks/Specs/acceptance.json) for acceptance gates.

> [!NOTE]
> **Candidate Status (`3.13.1`) & Provenance (`3.13.0`)**:
> - [`ACCURACY/3.13.1.md`](ACCURACY/3.13.1.md) documents a **frozen working-tree release candidate** (`branch: 3.13.1-native-models`, `HEAD: 8e45a1e88ca0f8ae072321fe7852d81aec6a6fe6`), **not yet a published Git tag**.
> - **`v3.13.0` has no independent classical/standardized benchmark run** on disk. [`ACCURACY/3.13.0.md`](ACCURACY/3.13.0.md) retains the audited `v3.12.0` classical baseline alongside `v3.13.0`'s packed LLM, binary-parser bounds, and embedding safety checks.

---

## Version Archive

| Version | Release Status | Validation & Conformance Scope | Report | Performance Companion |
| :--- | :---: | :--- | :--- | :--- |
| **`3.13.1`** | Candidate (`HEAD 8e45a1e`) | `certification` (`27/27`), `dataframe-conformance` (`24/24`), `scientific-workflow` (`4/4`), CSV null-token round-trip, `SwiftStats`/`SwiftPreprocessing` math fixes, ARIMA `R²` root cause | [**`ACCURACY/3.13.1.md`**](ACCURACY/3.13.1.md) | [`PERFORMANCE/3.13.1.md`](PERFORMANCE/3.13.1.md) |
| **`3.13.0`** | Released (`v3.13.0`) | Carries forward `3.12.0` classical baseline + packed 4-bit/8-bit LLM, parser bounds, thread-safe `AppleSentenceEmbeddingEngine` | [`ACCURACY/3.13.0.md`](ACCURACY/3.13.0.md) | [`PERFORMANCE/3.13.0.md`](PERFORMANCE/3.13.0.md) |
| **`3.12.0`** | Released (`v3.12.0`) | Audited `standard-3.12.0` / `migration-3.12.0`, deterministic FNV-1a text embeddings, GPU BCE gradient, wire-protocol hardening | [`ACCURACY/3.12.0.md`](ACCURACY/3.12.0.md) | [`PERFORMANCE/3.12.0.md`](PERFORMANCE/3.12.0.md) |
| **`3.11.0`** | Released (`v3.11.0`) | Audits 1–15 mathematical remediations (TreeSHAP/KernelSHAP/LIME, ETS/Kalman, DecisionTree/LinearRegression, TargetEncoder, WordNet/VADER) | [`ACCURACY/3.11.0.md`](ACCURACY/3.11.0.md) | [`PERFORMANCE/3.11.0.md`](PERFORMANCE/3.11.0.md) |
| **`3.10.3`** | Released (`v3.10.3`) | Blocked centered ANOVA, Kahan-Babuška dot products, PCA total-variance EVR, Apache Parquet spec conformance | [`ACCURACY/3.10.3.md`](ACCURACY/3.10.3.md) | [`PERFORMANCE/3.10.3.md`](PERFORMANCE/3.10.3.md) |

---

## `3.13.1` Candidate Certification & Audit Summary

| Check / Suite | Scope | Verified Result |
| :--- | :--- | :--- |
| **`certification`** (`bench.py`) | 27 NIST StRD statistical cases | **27/27 PASSED**, max SwiftSci abs error `5.588e-09`, `certificate.json`: `"passed"` |
| **`dataframe-conformance`** (`bench.py`) | 24 DataFrame gather/filter/sort/null/join/group checks | **24/24 PASSED**, max abs error `0.000e+00`, `certificate.json`: `"passed"` |
| **`scientific-workflow`** (`bench.py`) | 4 end-to-end scientific workflows | **4/4 PASSED**, max abs error `1.421e-14`, `certificate.json`: `"passed"` |
| **Package Test Suite + Cookbook** | 15 Swift modules (Debug & Release) + 70 recipes | **`1,833` passed** (`138` XCTest + `1,695` Swift Testing), `0` failed, `3` optional external-model tests skipped; `CookbookVerification` **`70/70`** |
| **CSV Null-Token Round-Trip** | `"NA"`, `"null"`, `"NaN"`, `"None"`, `""`, `nil` | Quoted tokens retain literal `String`; unquoted `""`/tokens decode as `nil`; writer outputs `2,290,455 B` (`+124,999 B` quotes vs unquoted Pandas `2,165,456 B`) |
| **`TypedColumn.unique` (`248` vs `247`)** | Nullable `[Double?]` canonical deduplication | **Intended contract**: SwiftSci keeps SQL `nil` distinct from IEEE `NaN` (`1 nil + 1 NaN + 1 ±0.0 + 245 = 248`); NumPy/Pandas collapses `nil` & `NaN` (`247`) |
| **`ARIMA(1,1,1)` Out-of-Sample `R²`** | `AccuracyBenchmarks.swift` (`N=500`, `period=12`, `h=24`) | **No code regression** vs `3.12.0` (`ARIMA.swift` unchanged). Both SwiftSci (`RMSE=10.218, R²=-1.555`) and Statsmodels (`RMSE=22.158, R²=-11.014`) fail on seasonal data vs `Holt-Winters` (`R²=+0.997`) and `Seasonal Naive` (`R²=+0.463`). **Previous `"R² agrees"` summary was corrected.** |

---

## Navigation & Reproduction

- Read the complete `3.13.1` candidate findings (including statistical checksums, CSV contract table, `TypedColumn.unique` analysis, full ARIMA baseline comparison, and `AccuracyBenchmarks` console outputs) in [**`ACCURACY/3.13.1.md`**](ACCURACY/3.13.1.md).
- Reproduce conformance audits and legacy research accuracy checks:

```bash
# Audited conformance profiles:
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py audit Benchmarks/Runs/certification-run

# Legacy in-process research scorecard (unvalidated for certification):
swift run -c release SwiftSciBenchmarks --research --suite Accuracy
python3 Benchmarks/Python/accuracy_benchmarks.py --research
```
