# ANOVA from original decimal observations

Preserve small differences around a large offset before converting observations to `Double`.

## Choose the input path

Use `Stats.oneWayANOVA(groups:)` for existing `Double` arrays. It operates on their binary64 values. It cannot recover decimal digits lost before the call.

Use `Stats.oneWayANOVA(decimalGroups:)` when the original observations are decimal text:

```swift
let result = try Stats.oneWayANOVA(decimalGroups: [
    ["10000000000000000.1", "10000000000000000.2", "10000000000000000.3"],
    ["10000000000000000.4", "10000000000000000.5", "10000000000000000.6"]
])
// result.fStatistic is approximately 13.5.
```

Keep the source text from CSV, JSON strings or instrument output. Formatting an already converted `Double` as a string cannot restore the original decimal value.

## What the decimal path preserves

The method parses the observations as exact decimal values, subtracts the first observation from every group, then converts the residuals to `Double`. It uses one common origin because subtracting a different origin from each group would change the between-group variation. ANOVA's F statistic, degrees of freedom, p-value and effect size are invariant under a common translation in exact arithmetic.

Parsing and subtraction occur once per call. The method then uses the existing binary64 ANOVA implementation. It does not turn the statistical calculation into arbitrary-precision arithmetic. Residual conversion still rounds to binary64, and subsequent operations retain binary64's precision and range limits. A dataset with a wide dynamic range can still need a different representation or algorithm.

## Accepted decimal text

Inputs use ASCII digits, an optional leading sign, a period as the decimal separator, and an optional `e` or `E` exponent. Examples include `-0.2`, `.2`, `2.` and `2e-1`. Whitespace, grouping separators, trailing text, NaN and infinity are rejected.

After removing leading and trailing zeros, a nonzero input must have at most 38 significant decimal digits and a coefficient exponent between -128 and 127. For example, `12.30` normalizes to `123 × 10^-1`. Zero ignores its sign and scale. Swift integer digit arithmetic parses each coefficient and subtracts aligned coefficients without rounding. A residual with more than 38 significant decimal digits is rejected before binary64 conversion.

The method throws `StatsError.invalidInput` if parsing or common-origin subtraction would lose decimal information. Group-count, empty-group and zero-within-group-variance errors follow the existing ANOVA API. Decimal parsing and checked subtraction cost more than accepting preconverted arrays. Use this path when preserving the source decimal differences matters.

## Validation

The NIST benchmark suite keeps separate original-decimal and exact-binary64-input workloads. Original-decimal ANOVA fixtures retain NIST's source tokens as strings and call this method. Binary64 fixtures continue to call `oneWayANOVA(groups:)`. Both retain their declared reference answers and acceptance tolerances.
