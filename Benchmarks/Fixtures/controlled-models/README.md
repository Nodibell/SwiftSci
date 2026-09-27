# Controlled inference and stateful models

These original synthetic fixtures check public API behavior using supplied inputs and independently calculated answers. They use the repository license. The `controlled-conformance` profile includes these cases and the [search and explanation cases](../controlled-search/README.md).

Reconstruct the input, reference and inventory bytes with Python's standard library:

```sh
python3 Benchmarks/Tools/build_controlled_model_fixtures.py
```

The generator uses exact Fraction arithmetic for affine predictions, a one-cluster centroid and Kalman recurrences. Logistic probabilities use Decimal arithmetic at 80-digit precision. References retain the higher-precision values and bind the source and derived input hashes. Runtime workers do not import reference generation code.

## Contracts

- Fixed linear regression uses the public supplied-weight constructor, requests CPU execution and predicts without fitting. Outputs include the supplied bias, weights and every prediction.
- Fixed logistic regression also predicts without fitting. Outputs include bias, weights, both class probabilities and every class label. Class one requires probability strictly greater than 0.5. The zero-logit rows check that boundary.
- One-cluster KMeans requests CPU execution and checks the centroid, training labels, held-out labels and inertia. The exact solution is the arithmetic mean. This does not certify multi-cluster initialization or stopping rules. The Python comparator computes the direct solution, rather than imitating the iterative Swift algorithm.
- Kalman filtering uses fully specified transition and observation matrices, process and measurement covariance, initial state and observations. The scalar and constant-velocity cases check every filtered mean and covariance and one subsequent prediction. Covariance updates use the Joseph form. Singular innovations and smoothing are outside these fixtures.

Each timed model operation creates a fresh instance. Decoding is outside timing; construction, configuration, model computation and complete output materialization are inside. The profile performs one warmup and two measured samples, validating each result. Repetition checks that prior Kalman state does not leak into the next sample.

Run `prepare --profile controlled-conformance`, then `run --profile controlled-conformance` with the worker arguments from the benchmark README. Both tolerances are 1e-12. A failure remains a failure and prevents a passing conformance record. These timings are diagnostic and are not a formal performance baseline. Numerical conformance, model quality and performance are separate claims.
