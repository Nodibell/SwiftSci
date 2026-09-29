# Compact numerical batches

`PreparedNumericBatch` provides owned Double columns between a dataframe and numerical consumers. It stores each column in a contiguous Swift array and allocates a validity bitmap only when that column contains missing entries. It is an additive API. `TypedColumn.values`, the heterogeneous dataframe representation, and existing matrix APIs retain their contracts.

## Prepare and reuse numerical inputs

```swift
import SwiftDataFrame
import SwiftPreprocessing

let input = try frame.prepareNumericBatch(["temperature", "pressure"])
let selected = try input.selectingRows(selectedRowIndices)
var scaler = StandardScaler()
let scaled = try scaler.fitTransform(selected)
let matrix = try scaled.matrix(order: .columnMajor, missing: .reject)
```

The batch accepts the same dataframe column types as existing feature extraction: Double, Int64 and Bool. Int64 conversion has the existing Double precision limits. Bool maps to zero or one. Column order and repeated column names follow the request. Unsupported types and missing names throw before conversion.

A caller that already owns dense Double columns can construct a batch directly:

```swift
var input = try PreparedNumericBatch(
    columnNames: ["temperature", "pressure"],
    columns: [temperatures, pressures]
)
```

Construction retains arrays using Swift copy-on-write. It checks column counts and equal lengths. Dense NaNs are valid values. An empty column list represents zero rows; dataframe preparation can also preserve a nonzero row count with zero selected columns.

`selectingRows` accepts a precomputed filter selection or sort permutation. It validates indices, gathers values and validity, and preserves order and duplicates. Repeated selections compose `originalRowIndices`, which refers to positions in the dataframe at preparation time. This mapping is not a persistent row key. Identity mappings remain implicit until requested.

StandardScaler and MinMaxScaler accept batches directly. Keeping a batch between operations avoids reconstructing optional dataframe columns. Concrete scaler overloads and the existing dataframe scaling conveniences use this path internally. Generic `PreprocessingTransformer` adapters retain their existing dispatch and nested-matrix behavior for other conformers.

## Numerical and missing-value behavior

The batch distinguishes missing entries from valid NaNs. Indexed reads return `Double?`; `nullCount(inColumn:)` counts only missing entries. `columnValues(at:)` creates an optional array when a consumer needs it.

Matrix export replaces missing values with NaN by default, matching existing feature extraction. `.reject` rejects missing entries but permits valid NaNs. The scalers retain the legacy arithmetic behavior, including its treatment of NaN input. Scaler results contain valid numerical values, which can be NaNs. They do not preserve the input validity mask or perform imputation.

StandardScaler retains row-width Accelerate operations with a small reusable workspace. Replacing them with whole-column scalar or broadcast division changed finite rounding on the tested Apple silicon toolchain. The retained implementation matches the old finite results bit for bit without allocating nested rows. MinMaxScaler operates on contiguous columns and passes the same finite compatibility checks.

## Export to a numerical consumer

`matrix(order:)` returns `PreparedNumericMatrix`, with owned contiguous values, shape, column names and original row positions. Choose `.rowMajor` or `.columnMajor` to match the consumer. Multi-column export packs one output buffer. A single-column export shares the values through copy-on-write.

Use the resulting values with an array-taking Accelerate API, or use Swift's scoped buffer access for pointer-taking APIs. A borrowed pointer must not escape its closure. Matrix export does not create a GPU allocation or guarantee zero-copy access by MLX, Metal, NumPy or Arrow. It provides the owned CPU buffer and layout needed to build those adapters explicitly.

## Update fixed-size batches

```swift
let snapshot = input
try input.updateColumn(at: 0, rows: [4, 12, 4], values: [18.0, nil, 19.0])
```

Updates validate the column, edit counts and every row before changing values. Repeated indices use the last supplied value, so row four ends at 19. Invalid input leaves the batch unchanged. An empty edit list is a no-op for a valid column. Assigning a nonmissing value restores validity; removing the last missing entry releases the bitmap.

A batch update acquires each mutable buffer once. Swift copy-on-write preserves previous batch copies and exported matrices. Shared edits copy the affected value buffer and bitmap; they can also copy column metadata. Unedited value buffers remain shared. Separate copies can be edited by separate tasks because the batch has value semantics and is Sendable. Concurrent edits to the same variable still require caller synchronization.

This API supports fixed-size indexed edits. Appending rows, changing schema, persistent views, generic element types and borrowed foreign buffers remain separate design work under [issue #40](https://github.com/Nodibell/SwiftSci/issues/40).

`payloadByteCount` reports logical value and validity bytes only. It excludes spare capacity, column metadata, row mappings and retained snapshots. Process memory measurements include these other costs.

The implementation keeps Swift tools version 6.0 and minimum macOS 14 unchanged. Validation and performance evidence come from Apple silicon on the recorded newer toolchain; they do not certify a separate Swift 6.0 compiler run.
