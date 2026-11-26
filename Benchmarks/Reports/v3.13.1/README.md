# SwiftSci 3.13.1 Historical Audit & Reproducibility Index

> **Status — Historical Snapshot (`v3.13.1`, Superseded by [`v3.13.2`](../v3.13.2/README.md))**: Preserves the raw audited measurements and verification details recorded on the `3.13.1` working tree (`HEAD: 8e45a1e88ca0f8ae072321fe7852d81aec6a6fe6`, generated `2026-11-14T20:12:12Z`) prior to the `v3.13.2` correctness fixes and retirement of legacy `--research` harnesses.

> [!WARNING]
> **v3.13.2 Audit Notice**: The `v3.13.1` codebase contained scientific defects remediated or formally scoped in `v3.13.2` (**G-033**, **G-037**, **G-035**, **G-036**, and `Stats.skewness`/`Stats.kurtosis` `ddof=1` documentation), and the legacy `--research` harnesses (`SwiftSciBenchmarks --research`, `Benchmarks/Python/benchmarks.py`, `Benchmarks/CSVAcceleration/`) were retired and removed in `v3.13.2`. See [v3.13.2 Evidence Index](../v3.13.2/README.md).

## Audit Artifacts

- [3.13.1 Deep Accuracy & Conformance Audit](accuracy.deep.md) — full `certification` (`27/27`), `dataframe-conformance` (`24/24`), `scientific-workflow` (`4/4`), CSV null-token byte-parity proof, `TypedColumn.unique` contract analysis, ARIMA `R² = -1.555` root-cause comparison, and retired `--research` console metrics.
- [3.13.1 Deep Performance Audit](performance.deep.md) — full environment provenance, OS/toolchain variance investigation (`comparison-suite-20261114-2033`), `standard` (`14` cases) and `migration` (`36` cases) `Mean ± SD / CV% / IQR / z-score` dumps, targeted `3.13.1` patched-path microbenchmarks (`21` cases), and retired `--research` archive tables.

## Companion Reports

- [Runner guide](../../README.md) · [Performance Overview](../../../PERFORMANCE/3.13.1.md) · [Accuracy Overview](../../../ACCURACY/3.13.1.md) · [Latest (`3.13.2`)](../v3.13.2/README.md)
