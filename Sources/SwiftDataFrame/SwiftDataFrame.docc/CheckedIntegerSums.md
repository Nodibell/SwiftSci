# Checked integer group sums

Use `sumChecked()` when integer totals must remain exact. Existing `sum()`, `agg` and `transform` calls keep their Double results and conversion behavior.

```swift
import SwiftDataFrame

let frame = try DataFrame(columns: [
    TypedColumn<String>(name: "account", values: ["a", "a"]),
    TypedColumn<Int64>(name: "amount", values: [9_007_199_254_740_993, 2])
])
let totals = try frame.groupBy("account").sumChecked()
let amounts = totals[column: "amount", as: Int64.self]
// amounts?.values == [9_007_199_254_740_995]
```

Int32, Int64 and Int columns return Int64 totals. The accumulator retains integer precision throughout addition. Intermediate totals may exceed Int64's range if later values bring the final total back into range. This works on the package's macOS 14 deployment target.

A final total outside Int64's range throws `SwiftMLError.integerOverflow(column:group:)`. The error identifies the source column and a zero-based group index in first-appearance order. No partial DataFrame is returned. Overflow does not wrap, become a missing value, or silently convert to Double.

Null values do not contribute to sums. A group with no non-null values returns nil. Key formatting, group order and missing-key behavior are the same as `sum()`.

Floating columns still return compensated Double sums. Their infinities, NaNs and intermediate overflow follow `sum()`; the checked API does not reject floating overflow. Custom columns declared as integer columns must expose Int32, Int64 or Int values. Other non-null values produce a type-mismatch error.

`sumChecked()` is opt-in. It does not change integer means, minima, maxima, or the existing `agg` and `transform` methods.
