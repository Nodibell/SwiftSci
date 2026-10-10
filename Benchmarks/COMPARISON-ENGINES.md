# Compare SwiftSci with Polars and DuckDB

Use the existing benchmark contracts and evidence audits to compare SwiftSci with the Python reference stack, Polars, and DuckDB. Start from the [standard local setup](README.md#run-locally). Run the following commands from the repository root.

Install `Python/requirements-comparison.txt` in the benchmark virtual environment. It preserves the existing pinned dependencies and adds Polars 1.44.2 and DuckDB 1.5.6. The optional engine tests report a skip in standard environments that lack either engine. In the comparison environment, all three must pass without skips. Run the controller tests, build a fresh uninstrumented Swift worker, then invoke:

```sh
Benchmarks/.venv-standardized/bin/python -m pip install -r Benchmarks/Python/requirements-comparison.txt
POLARS_MAX_THREADS=1 Benchmarks/.venv-standardized/bin/python -m unittest discover -s Benchmarks/Tests
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py build
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/comparison_suite.py \
  --swift-worker "$HOME/Library/Caches/SwiftSci/standardized-benchmarks/derived/Build/Products/Release/SwiftSciBenchmarkWorker" \
  --python "$PWD/Benchmarks/.venv-standardized/bin/python" \
  --output "$PWD/Benchmarks/Runs/comparison-01"
```

Use a new output directory for every run. The suite runs `standard`, `extended`, `public-data`, and `nist` with all four engines, the full `migration` profile with Swift/Python, and its explicit 19-case `tabular-migration` subset with all four. The remaining 17 migration cases are listed in the generated exclusions file. CPU qualification follows with the existing Swift/Python adapters.

The new engines use native Polars operations and DuckDB SQL, not pandas fallbacks. Matrix export uses NumPy to assemble the requested owned row-major output. The comparison suite defaults to `production-default`, which clears inherited thread overrides and uses native engine scheduling. It records the configured Polars and DuckDB pool sizes. The bounded `legacy-capped` mode used by conformance CI sets `POLARS_MAX_THREADS=1` before process start and `threads=1` on each DuckDB connection. DuckDB input preparation builds an in-memory table outside timing for operation-only cases. Ingestion and the complete wine pipeline include reading within timing. Both adapters materialize results before stopping the timer and validate complete values afterward. Polars sort retains tie order; DuckDB orders ties by the fixture's original row identifier.

DuckDB exports and scaling include explicit row ordering. Stable tie handling uses the fixtures' original-row ID convention; these adapters are not validated for arbitrary reordered input.

The `pandas` engine name denotes the existing mixed pandas/NumPy/SciPy/PyArrow worker. These libraries are not separate engines in this report. Reports retain failed cases and unresolved timings without computing speedups from them. Whole-process peak RSS includes runtime and validation memory. These are CPU comparisons. Native scheduling permits multicore execution; recorded pool sizes do not measure active thread use.

Implementation references: [Polars lazy materialization and ordering](https://docs.pola.rs/api/python/stable/reference/lazyframe/), [Polars thread configuration](https://docs.pola.rs/api/python/stable/reference/api/polars.thread_pool_size.html), [DuckDB Python API](https://duckdb.org/docs/current/clients/python/overview), and [DuckDB order preservation](https://duckdb.org/docs/lts/sql/dialect/order_preservation).

For a smaller check, use `bench.py run --profile tabular-smoke --engines polars,duckdb` after preparing that profile. Add `--python Benchmarks/.venv-standardized/bin/python` and a fresh `--output` directory, then audit that directory with `bench.py audit`. Benchmark conformance CI runs `tabular-smoke`, `public-smoke`, and `certification` for both engines. CI gates correctness and evidence integrity, not performance.
