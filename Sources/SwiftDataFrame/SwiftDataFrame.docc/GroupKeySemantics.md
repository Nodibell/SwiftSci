# Group key equality

`groupBy` compares each key column independently. A row belongs to an existing group only when every key component matches. Groups appear in the order of their first row, and `transform` expands the results back into the original row order.

Null is a distinct key value. Strings such as `"null"`, `"__null__"` and `"a||b"` retain their literal meaning. No string acts as a null marker or tuple separator. Integer keys keep their full precision. Date keys use Date equality, including subsecond differences.

Float and Double keys use these grouping rules:

- Positive and negative zero belong to the same group.
- All NaNs belong to the same group, regardless of sign or payload.
- Null remains distinct from NaN.
- Positive and negative infinity belong to different groups.

These rules apply to group identity. They do not change floating-point arithmetic or comparison operators elsewhere. Group key output remains String columns formatted from the first row of each group. For example, if negative zero appears first, the output label is `"-0.0"`. Different Date values may produce the same displayed label because formatting can omit subsecond detail; they still form separate groups.

## Custom columns

For a custom `AnyColumn`, built-in floating values follow the rules above. Other Hashable values use their equality and hash implementations together with their runtime type. A native Int and an Int64 exposed from the same custom column remain distinct types. Equal hash values alone never merge unequal keys.

`AnyColumn` does not require its values to support equality. Non-Hashable custom values therefore retain a compatibility fallback that compares their runtime type and string description within each key component. If unequal values of the same type have identical descriptions, this fallback groups them together. Use Hashable values when exact custom-key grouping is required.

## Compatibility

This corrects earlier collisions between null and literal sentinel strings, between delimiter-containing tuples, and between values whose text omits identity details. Signed zeros now share a group. Group output order and String key columns remain unchanged.

An empty key selection still produces no groups. Missing key names are still ignored; if all requested names are absent, the rows form one group. These historical behaviors are separate from key equality.
