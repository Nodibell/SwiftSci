# SwiftSci Performance Benchmarks Index

> **Versioned Documentation Architecture**: Full release-specific benchmark tables, methodology records, and raw environment metadata live under [`PERFORMANCE/`](PERFORMANCE/) paired with [`ACCURACY/`](ACCURACY/). Use [`Benchmarks/README.md`](Benchmarks/README.md) to run and audit standardized suites.

> [!NOTE]
> **Candidate Status (`3.13.1`) & Provenance (`3.13.0`)**:
> - [`PERFORMANCE/3.13.1.md`](PERFORMANCE/3.13.1.md) documents a **frozen working-tree release candidate** (`branch: 3.13.1-native-models`, `HEAD: 8e45a1e88ca0f8ae072321fe7852d81aec6a6fe6`, benchmarked `2026-11-14`), **not yet a published Git tag**.
> - **`v3.13.0` has no independent classical/standardized benchmark run** on disk (`Benchmarks/Runs/` contains `3.11.0`, `3.12.0`, and `2026-11-14` runs). [`PERFORMANCE/3.13.0.md`](PERFORMANCE/3.13.0.md) carries forward the audited `v3.12.0` baseline alongside `v3.13.0`'s `SwiftLLM` / `SwiftNLP` changes.

---

## Version Archive

| Version | Release Status | Benchmark Methodology & Profile Scope | Report | Accuracy Companion |
| :--- | :---: | :--- | :--- | :--- |
| **`3.13.1`** | Candidate (`HEAD 8e45a1e`) | Audited `standard`, `migration`, `certification`, `dataframe-conformance`, `scientific-workflow`, idle patched microbenchmarks (`2026-11-14T20:12:12Z`), and `--research` | [**`PERFORMANCE/3.13.1.md`**](PERFORMANCE/3.13.1.md) | [`ACCURACY/3.13.1.md`](ACCURACY/3.13.1.md) |
| **`3.13.0`** | Released (`v3.13.0`) | Carries forward `3.12.0` audited baseline (`SwiftLLM` / `SwiftNLP` changes only; no independent classical run) | [`PERFORMANCE/3.13.0.md`](PERFORMANCE/3.13.0.md) | [`ACCURACY/3.13.0.md`](ACCURACY/3.13.0.md) |
| **`3.12.0`** | Released (`v3.12.0`) | Audited `standard-3.12.0` & `migration-3.12.0` (`macOS 27.2`, Swift `6.4`) + legacy research tables | [`PERFORMANCE/3.12.0.md`](PERFORMANCE/3.12.0.md) | [`ACCURACY/3.12.0.md`](ACCURACY/3.12.0.md) |
| **`3.11.0`** | Released (`v3.11.0`) | `standard-3.11.0` (`macOS 27.0.1`, Swift `6.4`) + 15-audit ML/Data `--research` baseline | [`PERFORMANCE/3.11.0.md`](PERFORMANCE/3.11.0.md) | [`ACCURACY/3.11.0.md`](ACCURACY/3.11.0.md) |
| **`3.10.3`** | Released (`v3.10.3`) | Methodology v2 initial `DenseColumn` / compensated precision / 1M DataFrame history | [`PERFORMANCE/3.10.3.md`](PERFORMANCE/3.10.3.md) | [`ACCURACY/3.10.3.md`](ACCURACY/3.10.3.md) |

---

## `3.13.1` Candidate Highlights (`2026-11-14` vs `3.12.0` & Python)

All timings below are Apple M4 Pro (`arm64`, 24 GiB, Release `-O -whole-module-optimization`). Profiles are kept strictly separate:

- **Dedicated `.inner` Compact SIMD Hash Join (`migration`: `inner-join-v1`)**: **`2.185 ms`** std (`-79.5%` vs `3.12.0` `10.635 ms`) and **`2.209 ms`** mixed (`-84.0%` vs `13.808 ms`) vs Pandas `0.924 / 0.943 ms` (`0.42–0.43×`).
- **Mixed-Storage Filter (`migration`: `mixed-medium-filter`)**: **`0.2984 ms`** (`-27.9%` vs `3.12.0` `0.4138 ms`, **2.97× faster than Pandas** `0.8857 ms`).
- **Pre-Reserved Single-Buffer `CSVWriter` (`100k × 3`, Idle Microbenchmarks)**:
  - Numeric: **`9.878 ms`** (`1,552,950 B`, CV `3.5%`, **4.94× faster** than Pandas `48.797 ms`)
  - Null-Token Strings (`"NA"`, `"null"`, `"NaN"`, `"None"`, `""`, `nil`): **`15.714 ms`** (`2,290,455 B`, CV `3.0%`, **3.64× faster** than Pandas `57.223 ms`)
  - Plain Strings: **`17.528 ms`** (`2,830,955 B`, CV `2.3%`, **3.45× faster** than Pandas `60.423 ms`)
- **CSV Reads — Two Distinct Workloads**:
  - `medium-csv-read` ([`csv-read-v2.json`](Benchmarks/Specs/workloads/csv-read-v2.json), `100k × 5` **complete finite non-null schema**, no quoting): **`2.798 ms`** in `standard-3.13.1` (`2.366 ms` in `comparison-suite-20261114-2033`, **4.10× faster** than Pandas `11.474 ms`, `39.4` vs `215.9 MB` RSS).
  - `CSV_Read_Eager_100k_NullTokens` / `CSV_Read_Stream_100k_NullTokens` (`100k × 3` **quoted null-token round-trip**): **`3.512 ms`** (**3.02×** vs Pandas `10.618 ms`) and **`3.326 ms`** (**3.92×** vs Pandas `13.044 ms`).
- **Statistical & Preprocessing Kernels (Idle Microbenchmarks)**: `Stats_ChiSquare_GOF_100` **`0.0014 ms`** (**65.67×** vs SciPy), `Stats_ShapiroWilk_100` **`0.0033 ms`** (**20.02×**), `Stats_KS_Lilliefors_200` **`0.0072 ms`** (**11.52×**), and `RFE_1000x50_k10_step2` **`0.2592 ms`** (**62.40×** vs Scikit-Learn). Stricter two-pass/OOF paths remain slower than vectorized C/Python (`TargetEncoder_OOF` `24.243 ms` `0.38×`, `TypedColumn_Unique` `5.185 ms` `0.04×`, `DataFrame_Unique` `50.534 ms` `0.06×`).

---

## Profile Traceability & Environment Caveats

1. **Hash Join Is Not Comparable Across Different Profiles**:
   - Historical `3.11.0` `--research`: `2.701 ms` (`100k` rows)
   - Standardized `migration` (`inner-join-v1`, `bench.py`): `10.635 → 2.185 ms` std / `13.808 → 2.209 ms` mixed vs Pandas `0.924 / 0.943 ms`
   - `3.13.1` `--research`: `2.183 ms` median (`2.240 ms` mean) vs Pandas `0.4249 ms`
2. **Cross-Version OS/Toolchain Shift (`3.12.0` vs `3.13.1`)**: `3.12.0` ran on `macOS 27.2 (26B123)` / Swift `6.4` / Xcode `27.1`; `3.13.1` ran on `macOS 27.2.1 (26B131)` / Swift `6.4.2` / Xcode `27.2`. Unmodified Python/Pandas (`Pandas 3.0.6`, `NumPy 2.5.3`, `SciPy 1.18.1`, `PyArrow 25.0.1`, Python `3.14.6`) slowed down on `5/14` `standard` and `19/36` `migration` cases on the same host, while the flagged Swift kernels have zero source diff and recovered in `comparison-suite-20261114-2033` (`medium-csv-read = 2.366 ms`, `medium-sort = 2.270 ms`, `medium-group-sum = 0.390 ms`, `medium-minmax-scale = 0.792 ms`).
3. **Python Version Recording in `standard-3.13.1`**: `plan.environment.python` in `standard-3.13.1/run.json` recorded controller Python `3.11.9`, whereas `plan.engines.pandas.version` recorded the actual Pandas worker in `.venv-standardized` (`Python 3.14.6`). Both `migration-3.13.1` and `comparison-suite-20261114-2033` used Python `3.14.6` throughout.

---

## Migration Note & Quick Reproduction

Prior monolithic history from `v3.10.3` through `v3.13.0` is preserved verbatim in `PERFORMANCE/<version>.md` with subdirectory-relative links; detailed `3.13.1` candidate tables live in [`PERFORMANCE/3.13.1.md`](PERFORMANCE/3.13.1.md).

```bash
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py inventory
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py run \
  --profile standard --engines swiftsci,pandas --output Benchmarks/Runs/standard-run
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py compare \
  Benchmarks/Runs/standard-3.12.0 Benchmarks/Runs/standard-run
```
