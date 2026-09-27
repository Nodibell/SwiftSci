# Exact model fixtures

These original synthetic fixtures check CPU numerical behavior. They do not measure held-out predictive quality. `source-spec.json` defines every training row, query row and exact rational constant. The source uses the repository license.

Rebuild with Python's standard library from the repository root:

```sh
python3 Benchmarks/Tools/build_exact_model_fixtures.py
```

The generator verifies the PCA eigensystem and Naive Bayes priors and likelihoods with exact `Fraction` arithmetic. It writes inputs separately from answers and binds each reference to source and input hashes. `rationalValues` retains exact fractions; `values` contains the binary64 comparison values. Runtime workers do not read the reference while computing outputs.

## PCA

The two PCA cases use four observations, two features and three held-out query rows. One retains both components; the other retains one. The exact sample variances are `200/3` and `50/3`. Explained-variance ratios use total input variance, yielding `4/5` and `1/5`; retaining one component must retain the ratio `4/5`.

Outputs include the fitted mean, retained variances and ratios, each component's outer product, training and query score Gram matrices, and the training-to-query cross-Gram matrix. Each reference records segment shapes, offsets and lengths. The checks allow consistent eigenvector sign choices while detecting inconsistent signs between fitting and later transformation. The eigenvalues are distinct; this fixture does not establish behavior for repeated eigenvalues or rank-deficient data.

SwiftSci explicitly uses CPU full SVD. The Python comparator uses NumPy's symmetric eigensolver on sample covariance. The mathematical contract is shared; the algorithms differ. Model fitting, transformation, state export and materializing the complete checked output are timed. The Gram matrices make these small conformance timings unsuitable as PCA throughput benchmarks.

## Multinomial Naive Bayes

Labels deliberately use `10` and `20`, and the training order starts with `20`. The contract checks sorted classes, all query probabilities and the raw prediction indices. SwiftSci's public `predict` returns indices into sorted classes. An index is not a class label.

Both implementations use empirical priors and positive Laplace smoothing. The Python comparator computes counts and log-joint probabilities with NumPy and `scipy.special.logsumexp`; it does not call scikit-learn. The expected answers come from rational arithmetic, independently of both runtime implementations. The query set includes an all-zero row, whose posterior equals the prior.

## Run

Use `prepare --profile model-conformance`, then `run --profile model-conformance` with the worker options in the benchmark README. Both tolerances are `1e-12`. Every output element is validated. Failing cases remain in the profile and prevent a passing conformance record. This profile is diagnostic until all required numerical contracts pass; it is not a mandatory passing CI gate.
