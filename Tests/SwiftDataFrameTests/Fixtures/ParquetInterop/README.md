# Parquet interoperability fixtures

`pyarrow-25.0.1.parquet` was generated with PyArrow 25.0.1, Snappy, PLAIN values, data-page version 1 and no statistics. It contains ids 0 through 128, a null Boolean at every fifth row and alternating Boolean values elsewhere, plus `rowN` labels with a null at every seventh row. The values are synthetic and use the repository license.

`legacy-swiftsci.parquet` preserves output from the previous SwiftSci writer. Its nonstandard compressed headers and byte-per-Boolean layout are retained solely to test backward reading compatibility. It is not a standards-conformance fixture.

- `legacy-swiftsci.parquet` SHA-256 `a7cc661ae70b7d64a12ae7cd4492de1239202a5dfd03e77b332756b3deae2b24`
- `pyarrow-25.0.1.parquet` SHA-256 `e182dc8f4674fdfcfdb553199d31d265eddcbf5de228d4710db794948dcae153`
