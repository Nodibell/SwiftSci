# Benchmark results

Selected evidence belongs here with its raw records, source identity and a README explaining scope. Keep ordinary local runs in ignored `Benchmarks/Runs/`.

Use the [standardized runner](../README.md) for new comparisons and conformance records. Existing files remain historical evidence under their original measurement rules. They do not acquire certification when a workload migrates.

## Production-default baseline

The [October 1, 2026 baseline](../Baselines/ProductionDefault-20261001/README.md) compares SwiftSci with pandas/NumPy, Polars and DuckDB using native threading defaults. It retains 10,300 samples, resolved experiment metadata and an export verifier. Treat it as an initial desktop measurement, not a CI performance threshold. CPU regression policy passed with four known Python conformance failures retained.

## Compare standardized runs

Build, run and audit each revision as described in the runner guide, then compare compatible run directories:

```bash
python3 Benchmarks/Tools/bench.py compare Benchmarks/Runs/baseline Benchmarks/Runs/candidate
```

The legacy `Python/compare.py` command is retired. Its name matching, zero-duration ratios and CI gates cannot establish a valid comparison.

## Legacy `--research` runners (retired in v3.13.2 cleanup)

The legacy unvalidated `--research` executables and ad-hoc result JSONs (`SwiftSciBenchmarks`, `Benchmarks/Swift/`, `Benchmarks/Python/benchmarks.py`, `Benchmarks/Python/accuracy_benchmarks.py`, `Benchmarks/Python/compare.py`, `Benchmarks/generate_fixtures.py`, `Benchmarks/CSVAcceleration/`, `Benchmarks/Results/python_results.json`, `Benchmarks/Results/swift_results.json`, and root `python_accuracy.json`) were removed from the active workspace during the v3.13.2 benchmark cleanup. See [DataFrameOptimization/README.md](DataFrameOptimization/README.md) for the hash-locked archive note (its `artifact-hashes.json` manifest remains intact), and use [Tools/bench.py](../Tools/bench.py) or `Benchmarks/Tools/scientific_validation_suite.py` for reproducible, contract-verified benchmarks.

Kiraa results under `DataFrameOptimization` refer to an unofficial development build with known bugs. Treat that build as an optional experimental reference, never an accuracy oracle or production baseline.
