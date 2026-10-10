# Float16 Core ML fixtures

These weight-free ML Program models apply elementwise ReLU to Float16 input and
return Float16 output. `matrix.mlmodel` has shape `[3, 2]`. `vector.mlmodel` has
shape `[2]`. Both use the names `features` and `result`.

The fixtures contain no training data or downloaded model weights. Their operation
has an independent expected result, `max(0, Float16(input))`. Tests exercise column
ordering, source-row identity, conversion overflow, byte accounting, batching and
pooled concurrency through the existing predictor interfaces.

Swift tests compile the checked-in models with Core ML. They require no Python
installation or network access. The model format targets macOS 13, while the
adapter's typed Float16 access requires macOS 15. The availability test preserves
the adapter's rejection behavior on macOS 14.

To regenerate the fixtures, run `generate.py` in an environment with coremltools
9.0 and NumPy 2.0.2. The generator removes the conversion date and serializes the
protobuf deterministically. `manifest.json` records versions and file checksums.
