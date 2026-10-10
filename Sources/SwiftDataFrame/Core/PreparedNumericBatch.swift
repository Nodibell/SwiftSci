/// Owned Double columns for numerical preprocessing without nested row arrays.
/// Missing values have separate validity bits. Valid NaNs remain distinct from missing values.
/// Copies preserve snapshots through Swift array copy-on-write.
public struct PreparedNumericBatch: Sendable {
    public let columnNames: [String]
    public let rowCount: Int
    public var columnCount: Int { columns.count }
    package var columns: [CompactNumericColumn]
    fileprivate let sourceRows: [Int]?

    /// Row positions in the dataframe at preparation time, including repeated selections.
    public var originalRowIndices: [Int] { sourceRows ?? Array(0..<rowCount) }

    package init(columnNames: [String], columns: [CompactNumericColumn], rowCount: Int, sourceRows: [Int]? = nil) {
        precondition(columnNames.count == columns.count && columns.allSatisfy { $0.values.count == rowCount })
        precondition(sourceRows == nil || sourceRows!.count == rowCount)
        self.sourceRows = sourceRows
        self.columnNames = columnNames
        self.columns = columns
        self.rowCount = rowCount
    }

    /// Reads a value without materializing an optional array. Indices must be in bounds.
    public subscript(row: Int, column: Int) -> Double? { columns[column][row] }

    public func nullCount(inColumn column: Int) -> Int { columns[column].nullCount }

    /// Materializes an optional column for callers that require the existing array representation.
    public func columnValues(at column: Int) -> [Double?] {
        (0..<rowCount).map { columns[column][$0] }
    }

    /// Logical bytes in value and validity buffers, excluding capacity and metadata.
    public var payloadByteCount: Int {
        columns.reduce(0) { $0 + $1.values.count * MemoryLayout<Double>.stride
            + ($1.validity?.count ?? 0) * MemoryLayout<UInt64>.stride }
    }

    /// Applies a filter selection or sort permutation in the supplied order.
    /// Duplicate rows are retained. All indices are validated before allocating output.
    public func selectingRows(_ indices: [Int]) throws -> PreparedNumericBatch {
        guard indices.allSatisfy({ $0 >= 0 && $0 < rowCount }) else {
            throw SwiftMLError.invalidParameter("Row selection contains an out-of-bounds index")
        }
        let mapping = sourceRows.map { original in indices.map { original[$0] } } ?? indices
        return PreparedNumericBatch(columnNames: columnNames, columns: columns.map { $0.gathered(at: indices) },
                                    rowCount: indices.count, sourceRows: mapping)
    }

    /// Packs Double values once in the consumer's requested order.
    /// Rejecting missing values still permits valid NaNs. The default matches existing matrix extraction.
    public func matrix(order: NumericMatrixOrder = .rowMajor,
                       missing: NumericMissingValuePolicy = .nan) throws -> PreparedNumericMatrix {
        if missing == .reject, let index = columns.firstIndex(where: { $0.nullCount > 0 }) {
            throw SwiftMLError.invalidParameter("Column \(columnNames[index]) contains missing values")
        }
        let (count, overflow) = rowCount.multipliedReportingOverflow(by: columnCount)
        guard !overflow else { throw SwiftMLError.invalidParameter("Matrix element count overflows Int") }
        let output: [Double]
        if columnCount == 1 {
            output = columns[0].values
        } else if order == .columnMajor {
            var packed = [Double]()
            packed.reserveCapacity(count)
            for column in columns { packed.append(contentsOf: column.values) }
            output = packed
        } else {
            var packed = [Double](repeating: 0, count: count)
            for c in columns.indices {
                for r in 0..<rowCount { packed[r * columnCount + c] = columns[c].values[r] }
            }
            output = packed
        }
        return PreparedNumericMatrix(values: output, columnNames: columnNames, rowCount: rowCount,
                                     order: order, sourceRows: sourceRows)
    }

    /// Creates a batch from dense Double columns without optional-array expansion.
    /// NaNs are valid numerical values. Use indexed updates to mark missing entries.
    public init(columnNames: [String], columns: [[Double]]) throws {
        guard columnNames.count == columns.count else {
            throw SwiftMLError.dimensionMismatch(expected: columnNames.count, got: columns.count)
        }
        let rows = columns.first?.count ?? 0
        for column in columns where column.count != rows {
            throw SwiftMLError.dimensionMismatch(expected: rows, got: column.count)
        }
        self.init(columnNames: columnNames, columns: columns.map(CompactNumericColumn.init), rowCount: rows)
    }

    /// Applies indexed updates to one column. Repeated row indices use the last supplied value.
    /// Invalid input throws before changing any values. Copies and previously exported matrices
    /// remain snapshots; copy-on-write detaches only the edited column's buffers when shared.
    public mutating func updateColumn(at column: Int, rows: [Int], values: [Double?]) throws {
        guard column >= 0 && column < columnCount else {
            throw SwiftMLError.invalidParameter("Column update index is out of bounds")
        }
        guard rows.count == values.count else {
            throw SwiftMLError.dimensionMismatch(expected: rows.count, got: values.count)
        }
        guard rows.allSatisfy({ $0 >= 0 && $0 < rowCount }) else {
            throw SwiftMLError.invalidParameter("Column update contains an out-of-bounds row")
        }
        guard !rows.isEmpty else { return }
        columns[column].update(at: rows, with: values)
    }

    package func replacingNumericColumns(_ values: [[Double]], columnNames: [String]? = nil) -> PreparedNumericBatch {
        PreparedNumericBatch(columnNames: columnNames ?? self.columnNames, columns: values.map(CompactNumericColumn.init), rowCount: rowCount, sourceRows: sourceRows)
    }

    package func replacingColumns(in frame: DataFrame) throws -> DataFrame {
        var result = frame
        for column in columns.indices {
            result = try result.withColumn(columnNames[column], column: TypedColumn<Double>(
                name: columnNames[column], values: columnValues(at: column)))
        }
        return result
    }
}

package struct CompactNumericColumn: Sendable {
    package var values: [Double]
    package var validity: [UInt64]?
    package var nullCount: Int

    package init(_ values: [Double]) {
        self.values = values
        validity = nil
        nullCount = 0
    }

    package init<T>(_ source: [T?], transform: (T) -> Double) {
        values = [Double](repeating: .nan, count: source.count)
        validity = nil
        nullCount = 0
        for (index, value) in source.enumerated() {
            if let value {
                values[index] = transform(value)
            } else {
                if validity == nil {
                    validity = [UInt64](repeating: .max, count: (source.count + 63) / 64)
                }
                validity![index >> 6] &= ~(UInt64(1) << (index & 63))
                nullCount += 1
            }
        }
    }

    package func gathered(at indices: [Int]) -> CompactNumericColumn {
        var result = CompactNumericColumn(indices.map { values[$0] })
        if let validity {
            for (destination, source) in indices.enumerated() {
                if validity[source >> 6] & (UInt64(1) << (source & 63)) == 0 {
                    if result.validity == nil {
                        result.validity = [UInt64](repeating: .max, count: (indices.count + 63) / 64)
                    }
                    result.validity![destination >> 6] &= ~(UInt64(1) << (destination & 63))
                    result.nullCount += 1
                }
            }
        }
        return result
    }

    package mutating func update(at rows: [Int], with replacements: [Double?]) {
        // Move the bitmap out so a unique column does not create a second owner
        // before entering its one scoped mutable borrow.
        var words = validity
        validity = nil
        var updatedNullCount = nullCount
        if words == nil && replacements.contains(where: { $0 == nil }) {
            words = [UInt64](repeating: .max, count: (values.count + 63) / 64)
        }
        values.withUnsafeMutableBufferPointer { values in
            if words != nil {
                words!.withUnsafeMutableBufferPointer { bits in
                    for (position, row) in rows.enumerated() {
                        let mask = UInt64(1) << (row & 63)
                        let wasValid = bits[row >> 6] & mask != 0
                        if let value = replacements[position] {
                            values[row] = value
                            bits[row >> 6] |= mask
                            if !wasValid { updatedNullCount -= 1 }
                        } else {
                            values[row] = .nan
                            bits[row >> 6] &= ~mask
                            if wasValid { updatedNullCount += 1 }
                        }
                    }
                }
            } else {
                for (position, row) in rows.enumerated() { values[row] = replacements[position]! }
            }
        }
        nullCount = updatedNullCount
        validity = updatedNullCount == 0 ? nil : words
    }

    package subscript(_ index: Int) -> Double? {
        if let validity, validity[index >> 6] & (UInt64(1) << (index & 63)) == 0 { return nil }
        return values[index]
    }
}

/// Element order for a contiguous numerical matrix.
public enum NumericMatrixOrder: Sendable {
    case rowMajor
    case columnMajor
}

/// Treatment of missing entries when exporting numerical values.
public enum NumericMissingValuePolicy: Sendable {
    case nan
    case reject
}

/// A contiguous Double matrix with value ownership and original row positions.
/// Swift copy-on-write preserves this matrix if its source batch is later mutated.
public struct PreparedNumericMatrix: Sendable {
    public let values: [Double]
    public let columnNames: [String]
    public let rowCount: Int
    public var columnCount: Int { columnNames.count }
    public let order: NumericMatrixOrder
    private let sourceRows: [Int]?
    public var originalRowIndices: [Int] { sourceRows ?? Array(0..<rowCount) }

    fileprivate init(values: [Double], columnNames: [String], rowCount: Int,
                     order: NumericMatrixOrder, sourceRows: [Int]?) {
        self.values = values
        self.columnNames = columnNames
        self.rowCount = rowCount
        self.order = order
        self.sourceRows = sourceRows
    }

    /// Reads an element in either layout. Indices must be in bounds.
    public subscript(row: Int, column: Int) -> Double {
        precondition(row >= 0 && row < rowCount && column >= 0 && column < columnCount)
        return values[order == .rowMajor ? row * columnCount + column : column * rowCount + row]
    }
}


extension PreparedNumericBatch {
    /// Materializes the row-array representation required by legacy consumers.
    /// Missing values become NaN, matching matrix extraction.
    public func rowValues() -> [[Double]] {
        (0..<rowCount).map { row in columns.map { $0.values[row] } }
    }

    package func requireFinite() throws {
        guard columns.allSatisfy({ $0.nullCount == 0 && $0.values.allSatisfy(\.isFinite) }) else {
            throw SwiftMLError.invalidParameter("Regression input contains missing or nonfinite values")
        }
    }

    package func replacingRows(_ rows: [[Double]]) throws -> PreparedNumericBatch {
        guard rows.count == rowCount, rows.allSatisfy({ $0.count == columnCount }) else {
            throw SwiftMLError.invalidParameter("Prepared transformer must preserve row count and feature schema")
        }
        return replacingNumericColumns((0..<columnCount).map { c in rows.map { $0[c] } })
    }
}

/// Features and a target column sharing one row selection and snapshot.
/// Select or reorder this value before splitting features from targets.
public struct PreparedSupervisedBatch: Sendable {
    private let batch: PreparedNumericBatch
    private let targetIndex: Int

    /// Builds a supervised view without copying column values.
    /// Column names must be unique and include at least one feature and one target.
    public init(_ batch: PreparedNumericBatch, targetColumn: String) throws {
        guard batch.columnCount > 1, Set(batch.columnNames).count == batch.columnCount,
              let index = batch.columnNames.firstIndex(of: targetColumn) else {
            throw SwiftMLError.invalidParameter("Supervised batch requires unique feature and target names")
        }
        self.batch = batch
        targetIndex = index
    }

    /// Feature columns in their original order, excluding the target.
    public var features: PreparedNumericBatch {
        let indices = batch.columns.indices.filter { $0 != targetIndex }
        return PreparedNumericBatch(columnNames: indices.map { batch.columnNames[$0] },
            columns: indices.map { batch.columns[$0] }, rowCount: batch.rowCount,
            sourceRows: batch.sourceRows)
    }

    /// Target values in the same order as the feature rows. Missing values become NaN.
    public var targets: [Double] { batch.columns[targetIndex].values }

    /// Selects all feature and target columns together, preserving duplicate rows.
    public func selectingRows(_ indices: [Int]) throws -> PreparedSupervisedBatch {
        try PreparedSupervisedBatch(batch.selectingRows(indices), targetColumn: batch.columnNames[targetIndex])
    }
}
