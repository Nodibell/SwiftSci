# Numeric filtering

`filter(column:where:)` and `filterFast(column:where:)` select rows in their original order. Built-in Int, Int32, Int64, Float and Double columns use typed comparison loops. Type dispatch and scalar threshold classification happen before scanning rows.

Integer comparisons preserve exact integer values, including values above Double's consecutive-integer range. Mixed integer and floating comparisons use the actual represented values without rounding an integer into equality or truncating a fractional threshold. For example, Int64 value 9,007,199,254,740,993 is greater than Double threshold 9,007,199,254,740,992. Conversely, that Double value is less than Int64 threshold 9,007,199,254,740,993.

Float values widen exactly to Double for comparisons. A Double threshold between adjacent Float values is not rounded to Float. NaN follows ordinary floating comparison rules: equality and ordered comparisons are false, while inequality is true. Positive and negative zero compare equal.

Missing values fail every ordinary comparison, including inequality. Use `isNull` to select missing values. `isNotNull` includes NaN because NaN is a present floating value.

These rules correct the former fallback behavior for native integers, Float NaNs and mixed comparisons near precision limits. Custom column implementations and unsupported cross-type comparisons retain their existing fallback behavior.
