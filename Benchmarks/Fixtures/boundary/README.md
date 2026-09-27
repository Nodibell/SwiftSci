# Dataframe-to-tensor conformance and bounded sweeps

These original MIT fixtures test the public dataframe conversion APIs followed by MLX affine arithmetic. They do not certify training, fitted estimator state or pretrained models.

Six conformance cases combine narrow and wide non-square matrices with CPU Float32, CPU Float64 and GPU Float32. Rows have unique IDs, filter exclusions and sort ties. The feature order differs from source storage. Every result includes actual dimensions, dtype, row IDs, exported Double features and targets, MLX tensor values, predictions, residuals and a mutation-isolation result. Float64 cases contain low bits that Float32 cannot retain. Metal Float64 is explicitly rejected.

The worker changes copies of exported matrix and target arrays and replaces a source column functionally. It then checks the retained source, selected frame and tensors. This establishes the tested value-isolation behavior, not physical zero-copy storage or deep copying of arbitrary user-defined reference columns.

The independent oracle selects rows by scalar predicates and stable ordering, casts to the declared precision and evaluates affine expressions with 100-digit Decimal arithmetic. The runtime comparator uses pandas and NumPy, always on CPU. Absolute tolerance is 2e-5 for Float32 and 1e-12 for Float64; relative tolerance is zero so large row IDs cannot conceal alignment errors. Dataset inputs and references are hash-bound to their source lock. Full conformance answers are bundled; sweep references pin the binary64 answer digest and reconstruct the complete answer outside measurement.

## Size and timing contracts

The `boundary-sweep` profile has 54 cases: 128, 1024 and 8192 input rows; 8 and 64 columns; three supported device/precision pairs; and three timing stages. Descriptors use the versioned `dyadic-v1` recipe. The worker expands the descriptor before timing. Three quarters of the rows survive filtering. The largest source feature payload is 4 MiB. These are bounded infrastructure checks, not a machine-memory saturation experiment.

| Stage | Prepared before timing | Timed work and actual output |
| --- | --- | --- |
| conversion | Source dataframe | Filter, stable sort, flat and nested feature export, target export, tensor allocation, synchronization and complete conversion readback |
| prepared | Evaluated feature, weight, target and bias tensors | Affine computation, residuals, synchronization and complete prediction/residual extraction |
| pipeline | Decoded recipe arrays | Frame construction, conversion, computation, complete readback and isolation checks |

Output extraction remains inside timing. Construction and isolation make the pipeline broader than the sum of the other two stages. Do not subtract stage times to infer an unmeasured component.

Run `python Benchmarks/Tools/sweeps.py RUN_DIRECTORY` to produce a verified report. Process lifetime peak RSS includes imports, preparation and validation. Source feature payload bytes and selected tensor payload bytes describe logical arrays only. None is an operation allocation counter, and their differences do not measure copy overhead. GPU-requested Swift results use NumPy CPU as a mathematical comparator, not a matched-backend performance baseline.

Run `python Benchmarks/Fixtures/boundary/generate.py` with the pinned reference environment to reproduce inputs, reference digests, manifests and profiles. `boundary-cpu-conformance` is the ordinary CPU subset; `boundary-conformance` adds explicit GPU execution. The sweep profile requires an Apple silicon host with Metal access. Successful sweeps establish measurement coverage; the formal performance baseline follows production repairs.
