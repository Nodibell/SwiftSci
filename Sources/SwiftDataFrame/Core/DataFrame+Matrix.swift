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
