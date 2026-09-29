# Dataframe semantic fixtures

These original bounded fixtures use the repository's MIT license. They test public SwiftDataFrame behavior with independent scalar expected answers. They are API conformance cases, not NIST certification or representative performance workloads.

Run `python3 Benchmarks/Fixtures/dataframe/generate.py` to regenerate the inputs, references, manifests and `dataframe-conformance` profile. Generation imports no dataframe library. Each manifest pins the generator, input and reference hashes. The controller also recomputes the scalar reference before accepting a pinned answer.

## Covered contracts

- Int64 gather, selection, filtering, sorting and grouping retain exact values beyond 2^53 and at the signed limits.
- Missing cells fail ordinary filter predicates, including inequality. Present NaN follows IEEE comparison rules. `isNotNull` includes NaN.
- Sorting preserves ties and places missing cells last in both directions. NaN sorting is excluded because the API has no deterministic NaN ordering contract.
- Group keys compare each component independently. Null differs from literal `null` and `__null__`; separators do not concatenate keys. Groups follow first appearance. Output keys are String columns. Row counts include all rows; selected aggregate counts include present NaN but exclude missing cells.
- Empty gathers and filters produce a schema-less frame under the existing API contract.
- Functional replacement leaves the source, selected source column and retained built-in TypedColumn unchanged. This does not establish deep ownership for custom reference-type AnyColumn implementations.
- Nested and flat matrix exports preserve requested column order, including repeated columns. Flat exports use row-major order. Numeric and Bool nulls become NaN. Int64 conversion to Double follows binary64 rounding. A non-square case detects transposition. Empty feature selection preserves the source row count and has zero columns.

Matrix tests establish logical output layout. They do not establish physical strides, zero-copy conversion, Arrow compatibility, GPU behavior or allocation efficiency.

## Input and result encoding

Inputs store Int64 as canonical decimal strings. Float64 cells accept finite JSON numbers or the strings `NaN`, `+Inf` and `-Inf`. JSON null always means a missing cell. Bool and UTF-8 values retain their JSON types. Validators reject unknown fields, ragged columns, duplicate names, invalid operation parameters and out-of-range gather indices before execution.

Results use exact finite integer words transported as binary64 values. Comparison uses zero absolute and relative tolerance. The frame-list prefix is its frame count. Each frame begins with row and column counts. Each column contains its type code, UTF-8 name length and bytes, null count, then cells. Type codes are Int64 1, Float64 2, Bool 3 and UTF-8 4. A missing cell is the word 0; a present cell begins with 1.

Int64 payloads are high and low unsigned 32-bit words of their two's complement representation. Float64 payloads are the corresponding IEEE-754 words, with all NaN payloads canonicalized to `0x7ff8000000000000`. Signed zero and infinities retain their bits. Bool payloads are 0 or 1. String payloads are UTF-8 byte lengths and bytes. Every word is exactly representable in binary64, so the transport cannot silently round large integers into agreement.

Ordinary actions return the original and resulting frames. Replacement also returns the previously selected source column and retained column as separate frames. Grouping returns the source, row counts and selected present counts. Matrix cases return the encoded source followed by row and column counts, nested row-major values, flat values, target length and target values. Matrix values use Float64 word pairs without cell tags.

## Pandas comparison

The pandas adapter uses object columns to retain None separately from NaN and preserve exact Python integers. Group keys use nullable String and Int64 columns with `dropna=False` and `sort=False`. Explicit validity masks enforce the declared null predicate rules. Empty selections normalize to SwiftDataFrame's schema-less result, and group keys normalize to its String outputs.

These conversions make the comparison an API compatibility check. They are not a benchmark of pandas defaults. Frame construction, operations and encoding are timed together; small fixture timings must not be used as a performance baseline.
