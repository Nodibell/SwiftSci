# Empty row selections

Row selection preserves column names, column order, and column types even when the result has no rows. Every retained column then has zero values and a null count of zero. The source dataframe remains unchanged.

This contract applies to `gathered(at:)`, predicate and column filtering, `head`, `tail`, and `sample`. Repeating these operations on a zero-row dataframe preserves its schema. Row predicates are not called when no rows exist.

A zero-row result can still select columns, extract an empty feature matrix, or prepare an empty numeric batch. Unsupported numeric column types and missing column names retain their existing error behavior. Prepared numeric batches also preserve their schema when a row selection is empty.

## Compatibility

Earlier releases returned a dataframe with no columns for many empty row selections. Code that uses `columnNames.isEmpty` to detect zero matching rows must use `rowCount == 0` instead. Code that deliberately needs a dataframe without columns can use `DataFrame.empty`.

`DataFrame.empty` and `DataFrame()` continue to contain no rows and no columns. Dropping every column still removes the schema. This change applies to row selection; it does not redefine joins, grouping, or other operations that determine their own output columns.

The difference is visible in a no-match feature pipeline:

```swift
let selected = source.filterRows { _ in false }
let batch = try selected.prepareNumericBatch(["temperature", "pressure"])
// batch has zero rows and the two named numeric columns.
```

The two feature names must exist in the source and have types supported by numeric preparation. Zero matches no longer cause a missing-column error solely because the row selection was empty.
