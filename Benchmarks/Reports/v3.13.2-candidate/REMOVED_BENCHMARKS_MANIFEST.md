# Removed Benchmarks & Preserved Baseline Manifest — `v3.13.2-candidate`

- **Base Commit:** `0cf99f0470d0757d0a02bd3244c62d821ab144b0` (`v3.13.1`)
- **Inventory Check:** `bench.py inventory` (`135` legacy entries indexed in `Benchmarks/Specs/legacy-inventory.json`; `legacy_sources_retired: true`; `30` standardized replacement workloads mapped)

---

## 1. Deleted Tracked Legacy Files (`24` Files)

All files below were deleted from the active tracked repository tree (`git` status `D` vs `v3.13.1`):

### 1.1 Legacy CSV Acceleration Standalone C/Swift Harness (`5` files)
- `Benchmarks/CSVAcceleration/README.md`
- `Benchmarks/CSVAcceleration/build.sh`
- `Benchmarks/CSVAcceleration/classify.c`
- `Benchmarks/CSVAcceleration/main.swift`
- `Benchmarks/CSVAcceleration/run.sh`
*(Replacement: Standardized `bench.py` CSV & streaming CSV profiles `standard`, `migration`, `tabular-migration` using `SwiftSciBenchmarkWorker`.)*

### 1.2 Legacy Swift `--research` In-Process Harness (`12` files)
- `Benchmarks/Swift/AccuracyBenchmarks.swift`
- `Benchmarks/Swift/BenchmarkDataLoader.swift`
- `Benchmarks/Swift/BenchmarkMain.swift`
- `Benchmarks/Swift/BenchmarkSuite.swift`
- `Benchmarks/Swift/DataFrameBenchmarks.swift`
- `Benchmarks/Swift/ExplainBenchmarks.swift`
- `Benchmarks/Swift/ExtensionBenchmarks.swift`
- `Benchmarks/Swift/ForecastBenchmarks.swift`
- `Benchmarks/Swift/LLMBenchmarks.swift`
- `Benchmarks/Swift/MLBenchmarks.swift`
- `Benchmarks/Swift/StatsBenchmarks.swift`
- `Benchmarks/Swift/VisionBenchmarks.swift`
*(Replacement: Audited `Benchmarks/Worker/SwiftSciBenchmarkWorker` + `Benchmarks/Worker/ScientificSuiteWorkloads.swift` + `Benchmarks/Tools/scientific_validation_suite.py`.)*

### 1.3 Legacy Python Ad-Hoc Scripts & Stale JSON Outputs (`7` files)
- `Benchmarks/Python/accuracy_benchmarks.py`
- `Benchmarks/Python/benchmarks.py`
- `Benchmarks/Python/compare.py`
- `Benchmarks/generate_fixtures.py`
- `Benchmarks/Results/python_results.json`
- `Benchmarks/Results/swift_results.json`
- `python_accuracy.json`
*(Replacement: `Benchmarks/Tools/bench.py`, `Benchmarks/Tools/acceptance.py`, `Benchmarks/Tools/scientific_validation_suite.py`, `Benchmarks/Python/scientific_reference_worker.py`, `Benchmarks/Python/standard_worker.py`, `Benchmarks/Python/tabular_worker.py`.)*

---

## 2. Intentionally Preserved Historical & External Artifacts

1. **`Benchmarks/Results/DataFrameOptimization/` (Intact):** Historical DataFrame optimization studies and `artifact-hashes.json` were preserved without modification.
2. **`/Users/oleksiichumak/Developer/MCW/Benchmarks/measure_startup.py` (Untouched External Asset):** Verified present outside the SwiftSci repository and never modified or deleted.
3. **Archived Reports (`PERFORMANCE/3.10.3.md`–`3.13.1.md`, `ACCURACY/3.10.3.md`–`3.13.1.md`, `Benchmarks/Results/README.md`):** Preserved for historical traceability with explicit deprecation/retirement notices pointing to `3.13.2-candidate.md`.
