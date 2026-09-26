/// Owned Double columns for numerical preprocessing without nested row arrays.
/// Missing values have separate validity bits. Valid NaNs remain distinct from missing values.
/// Copies preserve snapshots through Swift array copy-on-write.
public struct PreparedNumericBatch: Sendable {
    public let columnNames: [String]
    public let rowCount: Int
    public var columnCount: Int { columns.count }
    package var columns: [CompactNumericColumn]
    private let sourceRows: [Int]?

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

    package func replacingNumericColumns(_ values: [[Double]]) -> PreparedNumericBatch {
        PreparedNumericBatch(columnNames: columnNames, columns: values.map(CompactNumericColumn.init), rowCount: rowCount, sourceRows: sourceRows)
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
