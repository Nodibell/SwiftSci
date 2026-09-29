# NIST univariate reference data

These nine files are unchanged copies of the [NIST StRD univariate datasets](https://www.itl.nist.gov/div898/strd/univ/homepage.html). Their original headers retain descriptions, references and certified answers. Each manifest pins the byte count, SHA-256, observation count and first data line. Git preserves the original line endings.

| Dataset | Observations | NIST difficulty |
| --- | ---: | --- |
| PiDigits | 5,000 | Lower |
| Lottery | 218 | Lower |
| Lew | 200 | Lower |
| Mavro | 50 | Lower |
| Michelso | 100 | Lower |
| NumAcc1 | 3 | Lower |
| NumAcc2 | 1,001 | Average |
| NumAcc3 | 1,001 | Average |
| NumAcc4 | 1,001 | Higher |

Each dataset checks mean, sample standard deviation and sample variance. Mean and standard deviation use NIST's published answers. Variance is derived by squaring the published standard deviation with decimal arithmetic before conversion to binary64; it is not an independently certified NIST value. The manifest's `certified` field retains that derived reference for compatibility with the benchmark record format.

The fixture tests verify that manifests match the original headers. They independently recompute means and dispersions with 60-digit decimal arithmetic and compare against the published decimal answers. Workers parse the declared data section, require the exact observation count, and reject malformed or nonfinite values. They do not regenerate missing observations or silently truncate extra values.

## Floating-point acceptance

The inputs are decimal text, while both measured engines compute in binary64. A reference check accepts an error no greater than `atol + rtol * abs(reference)`. Each manifest records the effective tolerance for each statistic, and the resolved run plan includes it.

The absolute input-scale budget `b` is twice the largest binary64 ULP among that dataset's values. Mean and standard deviation use `atol = b`; variance uses `atol = 2 * reference_stddev * b + b * b`. Relative tolerances are `1e-14` for mean, `1e-12` for standard deviation and `2e-12` for variance. This fixed policy allows for decimal-to-binary representation and reduction rounding across datasets. It is not a proof of an algorithm's error bound. In particular, the large-offset NumAcc cases permit fewer relative digits in their small dispersion values than the other cases.

The tests check this policy against the manifests. Do not widen tolerances in response to a failing implementation. Inspect the recorded absolute error and distinguish input representation, reference rounding and algorithm error first. Output hashes identify results; they do not replace numerical validation.

## Scope

The `certification` profile runs 27 checks once per selected engine. The `nist` profile measures the same cases with warmups and independent process repetitions. Neither includes autocorrelation, ANOVA, linear regression or nonlinear regression. Those need separate API mappings and validation contracts.

A passing certificate records conformance to these specific assertions and the run protocol. It is not NIST endorsement, third-party accreditation or a claim that all library functions meet every published reference digit. See [NIST usage guidance](https://itl.nist.gov/div898/strd/general/howto.html).
