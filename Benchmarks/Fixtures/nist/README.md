# NIST NumAcc4

`NumAcc4.dat` is the unchanged ASCII reference file downloaded from [NIST](https://www.itl.nist.gov/div898/strd/univ/data/NumAcc4.dat). Its original header preserves the data description, references and certified answers. The dataset manifest pins its byte count and SHA-256.

The certification profile checks the mean and sample standard deviation against the NIST values. Sample variance 0.01 is derived from the certified standard deviation 0.1; it is not a separate NIST certification claim. Decimal inputs are decoded as binary64. The workload uses an absolute tolerance of 1e-8 to account for binary representation and reduction rounding. It does not demand every printed decimal digit from every algorithm.

A passing SwiftSci certificate records conformance to these specific assertions and the run protocol. It is not NIST endorsement, third-party accreditation or a claim that all library functions have been certified. See [NIST usage guidance](https://itl.nist.gov/div898/strd/general/howto.html).
