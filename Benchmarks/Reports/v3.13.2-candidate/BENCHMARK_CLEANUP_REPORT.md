# SwiftSci v3.13.2-candidate Benchmark Cleanup Report

- **Status:** `CANDIDATE` (Stage A — AC Power Verified, Uncommitted Working Tree)
- **Base Commit:** `0cf99f0470d0757d0a02bd3244c62d821ab144b0` (`v3.13.1`, `dirty: true`)
- **Stage A Run Snapshot `tree_sha256`:** `5abd55baabb35be660ab990e97eb37bd009c9d8a955b09c447d290a083bdee30` (used for Stage A benchmark/validation binary builds prior to the 1-line DocC backtick fix in `Sources/SwiftExplain/SwiftExplain.docc/ShapleyAttributions.md`)
- **Final pre-release review source fingerprint:** `953e773629eabf731530b82edd6d74813e8d5f4b22bfec25b412ba9951eb4a9c` (four DocC pages and one whitespace-only edit in `ARIMA.swift`; benchmark executable inputs, scripts, and thresholds are unchanged apart from that source whitespace).
- **Hardware / OS / Toolchain:** `Oleksiis-MacBook-Pro.local` (`Apple M4 Pro`, `arm64`, `24 GiB`), `macOS-27.2.1` (`26B131`), `Apple Swift 6.4.2` / `Xcode 27.2 (27B52)`, AC power (`AC -> AC`)

---

## 1. Objectives & Summary of Cleanup

Prior to `v3.13.2`, SwiftSci contained both an audited standardized runner (`Benchmarks/Tools/bench.py` + `SwiftSciBenchmarkWorker`) and legacy in-process `--research` harnesses (`Benchmarks/Swift/`, `Benchmarks/Python/{benchmarks,accuracy_benchmarks,compare}.py`, `Benchmarks/CSVAcceleration/`) whose outputs were frequently mixed into top-level comparisons (`README.md`, `PERFORMANCE/`, `ACCURACY/`).

In `v3.13.2-candidate`, the repository underwent a complete cleanup:

1. **Retired and deleted obsolete `--research` sources and stale result dumps** (`24` tracked files removed; see [`REMOVED_BENCHMARKS_MANIFEST.md`](REMOVED_BENCHMARKS_MANIFEST.md)).
2. **Removed the `SwiftSciBenchmarks` executable target dependency on legacy `--research` runners** in `Package.swift` and retired unvalidated console accuracy/timing outputs.
3. **Preserved intentional historical/external baselines:**
   - Kept `Benchmarks/Results/DataFrameOptimization/` (`artifact-hashes.json` and historical DataFrame optimization study artifacts intact).
   - Kept `/Users/oleksiichumak/Developer/MCW/Benchmarks/measure_startup.py` completely untouched outside the repository.
   - Annotated historical reports (`PERFORMANCE/3.10.3–3.13.1.md`, `ACCURACY/3.10.3–3.13.1.md`, `Benchmarks/Results/README.md`) with clear supersession and retirement warnings rather than rewriting historical logs.
4. **Introduced an accuracy-gated scientific validation suite (`Benchmarks/Tools/scientific_validation_suite.py`):**
   - Paired native worker `Benchmarks/Worker/ScientificSuiteWorkloads.swift` + isolated Python worker `Benchmarks/Python/scientific_reference_worker.py`.
   - Enforces `5` independent worker processes × `5` measured samples per process + `2` warmup iterations (`median of process medians`, IQR dispersion).
   - Enforces mandatory numerical accuracy, finiteness, fail-closed (`expect_error`), coalition Shapley exactness, rolling-origin out-of-sample, and **G-037** `fittedValues`/`residuals` gates **before** any timing speedup is eligible.
   - Automatically suppresses speedups on sub-microsecond stages (`< 1000 ns`, `EXCLUDED_RESOLUTION_LIMITED_LT_1000NS`), non-comparable algorithmic pairs (`NON_COMPARABLE_ESTIMATOR_OR_WEIGHTS`), and expected-error cases (`EXCLUDED_EXPECTED_ERROR`).
   - Discloses Peak RSS explicitly as whole-process `ru_maxrss` (`whole_process_peak_rss_ru_maxrss`), avoiding misleading allocation claims.
5. **Removed stale `--research` scorecards from top-level documentation:** Updated `README.md`, `PERFORMANCE.md`, `ACCURACY.md`, `Benchmarks/README.md`, and created `PERFORMANCE/3.13.2-candidate.md` and `ACCURACY/3.13.2-candidate.md`.

---

## 2. Ledger & Coverage State (`bench.py inventory`)

`Benchmarks/Tools/inventory.py` and `Benchmarks/Specs/legacy-inventory.json` account for all `135` legacy workloads without deleting audit traceability:

- **Total legacy inventory entries:** `135`
  - `research`: `75`
  - `standardized-core`: `60`
- **By kind:** `timed: 111`, `accuracy-result: 22`, `untimed-diagnostic: 2`
- **Replacement workloads mapped:** `30`
- **Flag:** `legacy_sources_retired: true`

---

## 3. Audit Safeguards & Methodology Corrections Enforced

| Defect / Pitfall in Legacy Harness | Remediation in `v3.13.2-candidate` |
| --- | --- |
| In-process Swift loops (`50-500` iterations) without warmup/process isolation | `bench.py` isolated OS process batches + `scientific_validation_suite.py` `5` processes × `5` samples (+ `2` warmups) |
| Sub-microsecond operations (`100 ns–800 ns`) advertised with inflated speedups | Automatic suppression threshold at `< 1000 ns` (`RES < 1 us` / `EXCLUDED_RESOLUTION_LIMITED_LT_1000NS`) |
| Accuracy/correctness failures still allowed performance tables to claim speedups | `scientific_validation_suite.py` marks `all_gates_passed` only when all `39/39` workloads & `3/3` rolling-origin checks pass; `acceptance.py` strictly separates raw conformance from reviewed baseline policy |
| `statsmodels` Kalman MLE vs SwiftSci Hannan-Rissanen or `sklearn` L-BFGS vs GD framed as apples-to-apples speedups | Labeled `comparable_estimator: false` (`NON_COMPARABLE_ESTIMATOR_OR_WEIGHTS`) with separate matched-estimator baselines (`ts_arima_*_matched_hr`, `ts_hw_*_fixed`) |
| `SwiftLLM` decode compared against fabricated/external baseline or missing parity bound | Rebuilt as internal architectural ablation (`KV-Cache` vs `Full-Forward` in SwiftSci) with strict greedy `max_abs_logit_diff < 1e-4` gate (`<= 4.77e-07` measured) |
| Mixed OS/toolchain baseline (`3.12.0` on `26B123/Swift 6.4` vs `26B131/Swift 6.4.2`) | `bench.py compare` blocks mismatched environments; reports disclose OS/toolchain boundaries explicitly |
| Binary Release instrumentation / Metal fast-math leaks | Verified `coverage_instrumentation: "absent"`, Release `-O -whole-module-optimization`, `MTL_FAST_MATH=NO`, `MTL_MATH_MODE=SAFE`, `MTL_MATH_FP32_FUNCTIONS=PRECISE` |

---

## 4. Post-Edit Verification Summary

- **`swift test -c release` (`16` test targets):** `138` XCTests + `1712` Swift Testing tests passed (`0` failures), `1` conditional suite skipped (`"Pinned Llama tokenizer parity"`). Includes `SwiftForecastTests` (`139/139`), `SwiftStatsTests` (`121/121`), and `SwiftExplainTests` (`26/26`).
- **Python harness suite (`Benchmarks/Tests`):** `236` tests (`unittest`), `0` failures (`skipped=4`).
- **Doc coverage (`scripts/verify_doc_coverage.py`):** `100.00%` (`2551/2551` matched API declarations, no unresolved placeholders).
- **Final review DocC:** All 14 targets generated archives with `--warnings-as-errors` after repairing four links in SwiftNLP and SwiftAgent.
