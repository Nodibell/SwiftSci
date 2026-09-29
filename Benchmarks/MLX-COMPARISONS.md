# Run Python MLX comparisons

Use an Apple silicon Mac with Xcode selected. Start with the [standard benchmark setup](README.md#run-locally), then run these commands from the repository root.

## Install and verify the engine

The optional requirements pin Python MLX and its Metal package to 0.31.1. This matches the core version declared by Swift's pinned mlx-swift 0.31.6 dependency. Matching version numbers does not establish identical compiler options or binary builds; the run records Python package versions and the Swift source/package identity separately.

```sh
Benchmarks/.venv-standardized/bin/python -m pip install -r Benchmarks/Python/requirements-mlx.txt
Benchmarks/.venv-standardized/bin/python -m unittest discover -s Benchmarks/Tests
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py build
```

To exercise Metal in the adapter tests, explicitly enable it:

```sh
SWIFTSCI_MLX_TEST_GPU=1 Benchmarks/.venv-standardized/bin/python -m unittest discover -s Benchmarks/Tests -p test_mlx_workers.py
```

An unavailable GPU fails an explicitly requested GPU run. It does not select CPU or record a passing skip. Ordinary CI runs the CPU adapter tests and CPU conformance profiles.

## Compare existing contracts

Choose a fresh output directory for each invocation. For CPU decoder checks:

```sh
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py prepare --profile neural-cpu-conformance
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py run \
  --profile neural-cpu-conformance --engines swiftsci,pandas,mlx \
  --swift-worker "$HOME/Library/Caches/SwiftSci/standardized-benchmarks/derived/Build/Products/Release/SwiftSciBenchmarkWorker" \
  --python Benchmarks/.venv-standardized/bin/python \
  --output Benchmarks/Runs/mlx-decoder-cpu-01
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py audit Benchmarks/Runs/mlx-decoder-cpu-01
```

Repeat with `boundary-cpu-conformance` for dataframe conversion and affine arithmetic. On a host with Metal access, use `neural-conformance` and `boundary-conformance` for both devices. Use `boundary-sweep` to vary the existing row counts, feature widths, precision and conversion/prepared/pipeline stages. See the [decoder contract](Fixtures/neural/README.md) and [boundary contract](Fixtures/boundary/README.md).

The `mlx` engine rejects other operations during planning. It also rejects `neural-loader-conformance`: Python has no equivalent of SwiftSci's public weight-loading API. Direct fixed-weight evaluation must not stand in for a test of that loader.

## Interpret the evidence

The decoder creates fresh fixed MLX weight arrays and K/V cache state for every sample. It verifies every weight by readback and uses native MLX arithmetic for normalization, projections, causal attention, rotary positions, feed-forward layers and logits. Cached samples evaluate every chunk before recording cache length. Stream synchronization and complete output extraction happen before the timer stops. The independent 80-digit oracle and unchanged tolerances validate the outputs afterward.

Swift constructs a public TransformerDecoder with initial parameters before replacing them; the Python adapter constructs the fixed tensor state directly. These construction paths differ. The four-token fixtures and end-to-end timings establish diagnostic correctness, not steady-state inference throughput or language-binding overhead in isolation.

The boundary adapter uses pandas for filtering, stable ordering and aligned feature/target export, then MLX for tensor allocation and affine arithmetic. NumPy participates in host array storage, not as an arithmetic fallback. Prepared-stage tensor creation and evaluation happen before timing. Conversion and pipeline stages retain their declared preparation boundaries. Float64 executes on CPU only. Full output values and mutation-isolation checks follow the existing contract.

The `pandas` engine remains a CPU NumPy mathematical comparison for GPU-requested cases. The `mlx` engine honors the requested device, but shares the MLX backend family with Swift; agreement between those two implementations alone cannot prove backend correctness. Whole-process RSS includes imports, preparation and validation. It is not a device allocation counter or evidence of zero-copy conversion.

Larger decoder architectures, long contexts, tokenizer and stopping contracts, quantized inference, pretrained checkpoints, MLX-LM and llama.cpp comparisons remain separate follow-up work. The existing boundary sweep is bounded to 8,192 rows and 64 features. This addition does not certify language quality or memory-capacity limits.
