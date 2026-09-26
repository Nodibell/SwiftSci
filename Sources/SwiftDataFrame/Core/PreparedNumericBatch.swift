/// Owned Double columns for numerical preprocessing without nested row arrays.
/// Missing values have separate validity bits. Valid NaNs remain distinct from missing values.
/// Copies preserve snapshots through Swift array copy-on-write.
public struct PreparedNumericBatch: Sendable {
    public let columnNames: [String]
    public let rowCount: Int
    public var columnCount: Int { columns.count }
    package var columns: [CompactNumericColumn]

    package init(columnNames: [String], columns: [CompactNumericColumn], rowCount: Int) {
        precondition(columnNames.count == columns.count && columns.allSatisfy { $0.values.count == rowCount })
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

    package func replacingNumericColumns(_ values: [[Double]]) -> PreparedNumericBatch {
        PreparedNumericBatch(columnNames: columnNames, columns: values.map(CompactNumericColumn.init), rowCount: rowCount)
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

    package subscript(_ index: Int) -> Double? {
        if let validity, validity[index >> 6] & (UInt64(1) << (index & 63)) == 0 { return nil }
        return values[index]
    }
}
