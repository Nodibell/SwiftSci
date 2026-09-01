import Foundation

extension DataFrame {
    /// Extracts named columns as a row-major [[Double]] matrix.
    /// Throws `DataFrameError.castFailed` for any column that isn't Double, Int64, or Bool.
    public func toFeatureMatrix(_ columns: [String]) throws -> [[Double]] {
        for colName in columns {
            if !columnNames.contains(colName) {
                throw SwiftMLError.columnNotFound(colName)
            }
        }

        let rowCount = shape.rows
        var matrix = [[Double]](repeating: [Double](repeating: 0.0, count: columns.count), count: rowCount)

        for (colIdx, name) in columns.enumerated() {
            if let col = self[column: name, as: Double.self] {
                for row in 0..<rowCount {
                    matrix[row][colIdx] = col[row] ?? .nan
                }
            } else if let col = self[column: name, as: Int64.self] {
                for row in 0..<rowCount {
                    matrix[row][colIdx] = col[row].map(Double.init) ?? .nan
                }
            } else if let col = self[column: name, as: Bool.self] {
                for row in 0..<rowCount {
                    if let val = col[row] {
                        matrix[row][colIdx] = val ? 1.0 : 0.0
                    } else {
                        matrix[row][colIdx] = .nan
                    }
                }
            } else {
                throw SwiftMLError.castFailed(column: name, targetType: "Double")
            }
        }
        return matrix
    }

    /// Extracts a single named column as a [Double] target vector.
    public func toTargetVector(_ column: String) throws -> [Double] {
        let matrix = try toFeatureMatrix([column])
        return matrix.map { $0[0] }
    }

    /// Extracts named columns as a contiguous 1D row-major flat [Double] buffer.
    public func toFlatFeatureMatrix(_ columns: [String]) throws -> (flat: [Double], rows: Int, cols: Int) {
        for colName in columns {
            if !columnNames.contains(colName) {
                throw SwiftMLError.columnNotFound(colName)
            }
        }
        let rows = shape.rows
        let cols = columns.count
        var flat = [Double](repeating: 0.0, count: rows * cols)

        for (colIdx, name) in columns.enumerated() {
            if let col = self[column: name, as: Double.self] {
                for r in 0..<rows {
                    flat[r * cols + colIdx] = col[r] ?? .nan
                }
            } else if let col = self[column: name, as: Int64.self] {
                for r in 0..<rows {
                    flat[r * cols + colIdx] = col[r].map(Double.init) ?? .nan
                }
            } else if let col = self[column: name, as: Bool.self] {
                for r in 0..<rows {
                    if let val = col[r] {
                        flat[r * cols + colIdx] = val ? 1.0 : 0.0
                    } else {
                        flat[r * cols + colIdx] = .nan
                    }
                }
            } else {
                throw SwiftMLError.castFailed(column: name, targetType: "Double")
            }
        }
        return (flat: flat, rows: rows, cols: cols)
    }
}

