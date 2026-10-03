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

## Reproduce legacy research

Run from the repository root. These commands label outputs as unvalidated research, with no certification or production claims:

```bash
swift run -c release SwiftSciBenchmarks --research \
  --json Benchmarks/Results/swift_research.json
python3 Benchmarks/Python/benchmarks.py --research \
  --json Benchmarks/Results/python_research.json
python3 Benchmarks/Python/accuracy_benchmarks.py --research \
  --json Benchmarks/Results/accuracy_research.json
```

Use a new filename to preserve existing results. A common seed or similar operation name does not establish equivalent inputs, algorithms or timing boundaries. Learned-model and GPU results remain research unless a standardized workload independently validates their complete output.

Kiraa results under `DataFrameOptimization` refer to an unofficial development build with known bugs. Treat that build as an optional experimental reference, never an accuracy oracle or production baseline.
