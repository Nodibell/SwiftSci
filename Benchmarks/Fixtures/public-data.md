# Public dataset contracts

## H2O-derived grouping

These workloads independently implement queries 1 through 5 from [H2O db-benchmark at c7421051](https://github.com/h2oai/db-benchmark/blob/c7421051af7530951d16b1505158371aebc0d2c1/datatable/groupby-datatable.R#L31-L99). The upstream scripts use the [Mozilla Public License 2.0](https://github.com/h2oai/db-benchmark/blob/c7421051af7530951d16b1505158371aebc0d2c1/LICENSE). No upstream script is copied here. Our generator and adapters are independent implementations under the SwiftSci repository license.

| Query | Keys | Aggregates in validation order |
| --- | --- | --- |
| Q1 | id1 | sum v1 |
| Q2 | id1, id2 | sum v1 |
| Q3 | id3 | sum v1, mean v3 |
| Q4 | id4 | mean v1, mean v2, mean v3 |
| Q5 | id6 | sum v1, sum v2, sum v3 |

The nine-column input has string keys id1 through id3, integer keys id4 through id6, integer values v1 and v2, and Double values v3. There are no nulls or nonfinite values. The upstream queries remove missing values; these fixtures exercise only their finite, non-null subset. Queries 6 through 10, joins, upstream size scales and upstream random streams are outside this profile.

Generator version 1 uses a 32-bit linear congruential sequence starting at 0x243F6A88, with multiplier 1664525 and increment 1013904223. Each draw shifts right eight bits before reducing modulo the requested domain. The broadly uniform variant samples string and integer domains separately. This is a deterministic engineering distribution with modulo bias, not a statistical randomness claim. The hot-key variant replaces all six keys with zero in four of every five rows. The generator still draws those keys first, so both variants retain identical measures.

Tiny inputs contain 257 rows and 7 low-cardinality key levels. Medium inputs contain 100,000 rows and 100 low-cardinality levels. High-cardinality domains contain floor of rows divided by low-cardinality levels. Values v1 are integers 1 through 5, v2 are integers 1 through 15, and v3 are exact quarter fractions from 0 through 99.75. Manifests pin every byte. These choices differ from the original benchmark and prohibit direct comparison to published H2O scores.

Each timer starts with a prepared nine-column frame and ends with a retained grouped frame. Converting key labels to numeric suffixes and sorting the complete result happen outside timing. Keys precede aggregates in the canonical output. The reference uses Python dictionaries and Decimal arithmetic, independent of either dataframe engine.

## UCI red-wine preprocessing

[Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), Cortez, Cerdeira, Almeida, Matos and Reis, 2009, DOI [10.24432/C56S3T](https://doi.org/10.24432/C56S3T), is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The bundled [original red-wine CSV](uci/winequality-red.csv) came from [UCI](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv). It contains 1,599 rows, 11 physicochemical inputs and a quality score, with no missing values. The [source documentation](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names) describes collection and columns.

Original bytes: 84,199. SHA-256: `4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e`.

Preparation converts semicolon separators to commas, changes spaces in headers to underscores, uses LF line endings and adds a zero-based source-row ID. Original numeric strings and row order are preserved. The dataset manifest pins both the original and derived bytes; the original file is unchanged. These transformations are modifications under the attribution terms.

The timed workflow reads that derived CSV, selects quality >= 6, stably sorts by alcohol ascending, fits population standard scaling on the selected rows for all 11 features, and exports IDs, targets and the row-major matrix. It produces 855 rows. Equal alcohol values retain source order. Validation checks all IDs, targets and 9,405 matrix values against an independent 50-digit Decimal reference. The absolute and relative tolerances are both 1e-10. This is a descriptive preprocessing workload; fitting after target-based selection must not be presented as an unbiased machine-learning evaluation.

The timer includes CSV parsing and every transformation and export. Fixture derivation, expected-answer computation and validation are excluded. Filesystem reads use the existing warm-cache policy. The complete output remains alive until after timing. This workload exercises the existing SwiftSci APIs without changing the production library.
