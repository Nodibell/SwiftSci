import Foundation

extension DataFrame {
    /// Extracts named columns as a row-major [[Double]] matrix.
    /// Throws `SwiftMLError.castFailed` for any column that isn't Double, Int64, or Bool.
    /// - Parameters:
    ///   - columns: List of column names to select or transform.
    /// - Throws: `SwiftMLError` or `DataFrameError` if column lengths mismatch, names collide, or I/O fails.
    /// - Returns: 2D numerical matrix of shape `[N, P]`.
    public func toFeatureMatrix(_ columns: [String]) throws -> [[Double]] {
        let selected = try featureColumns(columns)
        var matrix = [[Double]](repeating: [Double](repeating: 0.0, count: selected.count), count: shape.rows)
        for (index, column) in selected.enumerated() {
            column.forEachValue { row, value in matrix[row][index] = value }
        }
        return matrix
    }

    /// Extracts a single named column as a [Double] target vector.
    /// - Parameters:
    ///   - column: Target column identifier.
    /// - Throws: `SwiftMLError` or `DataFrameError` if column lengths mismatch, names collide, or I/O fails.
    /// - Returns: Array of computed numeric values.
    public func toTargetVector(_ column: String) throws -> [Double] {
        try toFlatFeatureMatrix([column]).flat
    }

    /// Extracts named columns as a contiguous 1D row-major flat [Double] buffer.
    /// - Parameters:
    ///   - columns: List of column names to select or transform.
    /// - Throws: `SwiftMLError` or `DataFrameError` if column lengths mismatch, names collide, or I/O fails.
    /// - Returns: The computed (flat: [Double], rows: Int, cols: Int) result instance.
    public func toFlatFeatureMatrix(_ columns: [String]) throws -> (flat: [Double], rows: Int, cols: Int) {
        let selected = try featureColumns(columns)
        let rows = shape.rows
        let cols = selected.count
        var flat = [Double](repeating: 0.0, count: rows * cols)
        for (index, column) in selected.enumerated() {
            column.write(to: &flat, offset: index, stride: cols, count: rows)
        }
        return (flat: flat, rows: rows, cols: cols)
    }

    /// Prepares owned compact Double columns for numerical operations.
    /// Accepts the same Double, Int64 and Bool columns as matrix extraction.
    /// Int64 conversion follows Double precision; nil and valid NaN remain distinct.
    public func prepareNumericBatch(_ columns: [String]) throws -> PreparedNumericBatch {
        let selected = try featureColumns(columns)
        return PreparedNumericBatch(columnNames: columns, columns: selected.map { $0.compact() }, rowCount: shape.rows)
    }

    /// Extracts an owned contiguous Double matrix without intermediate compact columns.
    ///
    /// Accepts Double, Int64 and Bool columns in the supplied order, including duplicates.
    /// Int64 conversion follows Double precision. The result remains a snapshot if the
    /// dataframe changes. Use `prepareNumericBatch(_:)` when column-based preprocessing
    /// is needed before packing a matrix.
    ///
    /// - Parameters:
    ///   - columns: Feature names in output order.
    ///   - rows: Original dataframe row positions. Nil selects all rows; an empty array
    ///     selects none. Order and duplicate positions are preserved.
    ///   - order: Storage layout requested by the consumer. Defaults to row-major.
    ///   - missing: Converts nil to NaN by default. Rejecting nil still permits valid NaNs.
    /// - Throws: `SwiftMLError` for missing or unsupported columns, invalid row positions,
    ///   a matrix size overflow, or selected nil values when `missing` is `.reject`.
    /// - Returns: A matrix retaining the selected names and original row positions.
    public func prepareNumericMatrix(_ columns: [String], rows: [Int]? = nil,
                                     order: NumericMatrixOrder = .rowMajor,
                                     missing: NumericMissingValuePolicy = .nan) throws -> PreparedNumericMatrix {
        let selected = try featureColumns(columns)
        let sourceCount = rowCount
        var sequentialRows = true
        if let rows {
            var expected = rows.first ?? 0
            for row in rows {
                guard row >= 0 && row < sourceCount else {
                    throw SwiftMLError.invalidParameter("Row selection contains an out-of-bounds index")
                }
                if row != expected { sequentialRows = false }
                expected = row + 1
            }
        }
        let count = rows?.count ?? sourceCount
        let (elements, overflow) = count.multipliedReportingOverflow(by: selected.count)
        guard !overflow else { throw SwiftMLError.invalidParameter("Matrix element count overflows Int") }
        var values = [Double](repeating: 0, count: elements)
        // Row indices and total capacity are validated before the scoped write borrow.
        try values.withUnsafeMutableBufferPointer { output in
            if let rows {
                try writeFeatureMatrix(selected, to: output, rows: rows, order: order,
                                       missing: missing, names: columns, sequentialRows: sequentialRows)
            } else {
                try writeFeatureMatrix(selected, to: output, rows: 0..<sourceCount, order: order,
                                       missing: missing, names: columns, sequentialRows: sequentialRows)
            }
        }
        return PreparedNumericMatrix(values: values, columnNames: columns, rowCount: count,
                                     order: order, sourceRows: rows)
    }

    private func featureColumns(_ names: [String]) throws -> [FeatureColumn] {
        // Preserve missing-name precedence even when an earlier column has an unsupported type.
        for name in names where self[column: name] == nil {
            throw SwiftMLError.columnNotFound(name)
        }
        return try names.map { name in
            if let column = self[column: name, as: Double.self] { return .double(column.values) }
            if let column = self[column: name, as: Int64.self] { return .integer(column.values) }
            if let column = self[column: name, as: Bool.self] { return .boolean(column.values) }
            throw SwiftMLError.castFailed(column: name, targetType: "Double")
        }
    }
}

private func writeFeatureMatrix<Rows: RandomAccessCollection>(
    _ columns: [FeatureColumn], to output: UnsafeMutableBufferPointer<Double>, rows: Rows,
    order: NumericMatrixOrder, missing: NumericMissingValuePolicy, names: [String], sequentialRows: Bool
) throws where Rows.Element == Int {
    guard !rows.isEmpty && !columns.isEmpty else { return }
    let count = rows.count
    // Keep small matrices and irregular gathers in complete column passes.
    // Rejecting writes also remain column-first to preserve error precedence.
    guard order == .rowMajor && missing == .nan && sequentialRows && output.count > 2048 else {
        for (column, source) in columns.enumerated() {
            let offset = order == .rowMajor ? column : column * count
            let stride = order == .rowMajor ? columns.count : 1
            try source.writeSelected(to: output, offset: offset, stride: stride,
                                     rows: rows, missing: missing, name: names[column])
        }
        return
    }
    // Target 16 KiB of output per tile, or one row when it is wider.
    let blockRows = max(1, 2048 / columns.count)
    var start = rows.startIndex
    var outputRow = 0
    while start != rows.endIndex {
        let length = min(blockRows, rows.distance(from: start, to: rows.endIndex))
        let end = rows.index(start, offsetBy: length)
        for (column, source) in columns.enumerated() {
            let offset = order == .rowMajor ? outputRow * columns.count + column : column * count + outputRow
            let stride = order == .rowMajor ? columns.count : 1
            try source.writeSelected(to: output, offset: offset, stride: stride,
                                     rows: rows[start..<end], missing: missing, name: names[column])
        }
        outputRow += length
        start = end
    }
}

private enum FeatureColumn {
    case double([Double?])
    case integer([Int64?])
    case boolean([Bool?])

    func compact() -> CompactNumericColumn {
        switch self {
        case .double(let values): return CompactNumericColumn(values, transform: { $0 })
        case .integer(let values): return CompactNumericColumn(values, transform: Double.init)
        case .boolean(let values): return CompactNumericColumn(values, transform: { $0 ? 1 : 0 })
        }
    }

    func write(to output: inout [Double], offset: Int, stride: Int, count: Int) {
        // Keep strided writes in a direct loop so the compiler can optimize the flat layout.
        switch self {
        case .double(let values):
            for row in 0..<count { output[row * stride + offset] = values[row] ?? .nan }
        case .integer(let values):
            for row in 0..<count { output[row * stride + offset] = values[row].map(Double.init) ?? .nan }
        case .boolean(let values):
            for row in 0..<count { output[row * stride + offset] = values[row].map { $0 ? 1.0 : 0.0 } ?? .nan }
        }
    }

    func writeSelected<Rows: Collection>(to output: UnsafeMutableBufferPointer<Double>, offset: Int, stride: Int,
                                        rows: Rows, missing: NumericMissingValuePolicy, name: String) throws
    where Rows.Element == Int {
        switch self {
        case .double(let values):
            try write(values, transform: { $0 })
        case .integer(let values):
            try write(values, transform: Double.init)
        case .boolean(let values):
            try write(values, transform: { $0 ? 1 : 0 })
        }

        func write<T>(_ source: [T?], transform: (T) -> Double) throws {
            for (destination, row) in rows.enumerated() {
                if let value = source[row] {
                    output[destination * stride + offset] = transform(value)
                } else if missing == .reject {
                    throw SwiftMLError.invalidParameter("Column \(name) contains missing values")
                } else {
                    output[destination * stride + offset] = .nan
                }
            }
        }
    }

    func forEachValue(_ body: (Int, Double) -> Void) {
        switch self {
        case .double(let values):
            for (row, value) in values.enumerated() { body(row, value ?? .nan) }
        case .integer(let values):
            for (row, value) in values.enumerated() { body(row, value.map(Double.init) ?? .nan) }
        case .boolean(let values):
            for (row, value) in values.enumerated() { body(row, value.map { $0 ? 1.0 : 0.0 } ?? .nan) }
        }
    }
}
