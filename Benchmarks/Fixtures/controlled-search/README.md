# Controlled search and explanations

These original synthetic fixtures use the repository license. Reconstruct their inputs and references with:

```sh
python3 Benchmarks/Tools/build_controlled_search_fixtures.py
```

Each source specification has a pinned digest. References bind the source and input hashes, and retain exact rational answers. The generator uses Fraction arithmetic and exhaustive coalition enumeration, independently of both runtime workers.

## Cosine search

The distinct-score case checks selected IDs, scores and closest-first order. The complete-ties case allows either native order within an equal-score group. Workers validate descending scores before canonicalizing ties by input row ID outside timing. No fixture truncates through a tie boundary.

The small-norm case checks identical nonzero vectors with mathematical cosine one. Equal dimensions and nonzero vectors are required. These fixtures do not establish a policy for zero vectors or mismatched dimensions.

The Swift worker builds a VectorStore before timing. Search and result export are timed; final identity, order and tie validation occur afterward. The Python comparator evaluates all vectors and sorts the results. This is a correctness comparison between different algorithms, not a matched implementation performance claim.

## KernelSHAP

Two-feature affine and interaction models have exhaustive exact Shapley answers. The public API replaces missing features with background column means. The nonlinear fixture uses one background row, avoiding ambiguity between mean imputation and averaging predictions over a distribution.

Each output contains the independently evaluated baseline, both returned feature contributions and the prediction. The returned contributions must sum to prediction minus baseline. The baseline and prediction are adapter evaluations of the supplied model; they are not fields returned by KernelSHAP.

The profile excludes sampled explanations with more than two features and TreeSHAP background/coverage conventions. Those require separate contracts. These checks do not certify explanation quality for arbitrary learned models.

Use the `controlled-conformance` profile described in the [controlled model guide](../controlled-models/README.md). Every output is checked with absolute and relative tolerance 1e-12. Failing cases remain visible; timing samples are diagnostic only.
