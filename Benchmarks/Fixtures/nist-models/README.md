# NIST model fixtures

National Institute of Standards and Technology, Statistical Reference Datasets, NIST Standard Reference Database 140, https://doi.org/10.18434/T43G6C. Accessed September 27, 2026.

The `originals` directory retains the downloaded NIST ASCII files unchanged, including all headers, references, certified values, whitespace, and line endings. Individual source URLs, byte counts, and SHA-256 digests appear in `sources.lock.json`. Generated input JSON appears in `inputs`; certified values and independent numerical references appear separately in `references`.

License label: NIST StRD; original notices retained. NIST's [licensing policy](https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications) distinguishes Standard Reference Data from other NIST works.

## Reconstruction

From the repository root, use Python with `mpmath==1.4.1`:

```sh
python Benchmarks/Tools/build_nist_model_fixtures.py
```

The default rebuild uses only the local originals and verifies their locked SHA-256 digests. It reconstructs input JSON, published reference strings, independent 80-digit decimal references, and references calculated from the exact binary64 input values. OLS references include every fitted prediction. ANOVA p-values are independent calculations, not NIST-certified values.

Add `--fetch` to download missing originals from their recorded NIST URLs. Existing originals are never replaced. Downloads must match their locked digests. Use `--base-dir PATH` to rebuild a separate directory containing a copy of `sources.lock.json` and `originals`.

## Runtime checks

Two profiles run all eight linear least-squares datasets and all eleven balanced one-way ANOVA datasets. Both use the same input bytes and the same CPU API calls.

- `numerical-conformance` checks fitted coefficients, residual sum of squares and ANOVA F against NIST's published decimal answers. OLS predictions use the independently reconstructed decimal solution; NIST does not publish those predictions.
- `numerical-binary64` checks against 80-digit calculations on the exact binary64 input values. This diagnoses arithmetic accuracy after input conversion. Passing it does not establish agreement with the original NIST decimal answers.

Every case declares `atol=1e-10` and `rtol=1e-9` before execution. These are engineering acceptance thresholds, not a claim that every published digit is reproduced. The large offsets in SmLs07, SmLs08 and SmLs09 lose information during binary64 conversion. Their exact binary-input F values differ from the decimal answers even before runtime arithmetic error. Both profiles retain these cases and report failures. They never relax a threshold or exclude a difficult dataset based on measured results.

Run each profile with `prepare` and `run` as described in the benchmark README. A failed profile cannot produce a passing conformance record. These small-case timings are diagnostic only and do not support throughput claims. Numerical failures require investigation before adding these profiles as mandatory passing CI gates.

Input decoding and fixture validation occur before timing. OLS times public CPU fitting, in-sample prediction, coefficient export, residual calculation and output materialization. SwiftSci's CPU API can fall back from its analytical solver to gradient descent; the API exposes the resolved device but not the chosen solver. The Python adapter uses SciPy's pivoted QR `gelsy` driver. The suite compares the declared mathematical outputs, without claiming identical algorithms. ANOVA times the public operation and exports F and both degrees of freedom. Every coefficient and prediction is checked; a checksum alone is insufficient.
