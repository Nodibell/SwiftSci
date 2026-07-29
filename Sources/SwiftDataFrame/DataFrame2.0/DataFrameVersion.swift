import Foundation

/// Snapshot metadata for DataFrame versioning.
public struct DataFrameVersionSnapshot: Sendable, Codable, Equatable {
    /// The version.
    public let version: Int
    /// The tag.
    public let tag: String
    /// The timestamp.
    public let timestamp: Date
    /// The row count.
    public let rowCount: Int
    /// The column count.
    public let columnCount: Int

    /// Creates a new instance.
    /// - Parameters:
    ///   - version: The version.
    ///   - tag: The tag.
    ///   - timestamp: The timestamp.
    ///   - rowCount: The row count.
    ///   - columnCount: The column count.
    public init(version: Int, tag: String, timestamp: Date = Date(), rowCount: Int, columnCount: Int) {
        self.version = version
        self.tag = tag
        self.timestamp = timestamp
        self.rowCount = rowCount
        self.columnCount = columnCount
    }
}

/// Utility for side-by-side DataFrame diffing and metrics deltas.
public struct DataFrameDiff: Sendable {
    /// The row delta.
    public let rowDelta: Int
    /// The column delta.
    public let columnDelta: Int
    /// The added columns.
    public let addedColumns: [String]
    /// The removed columns.
    public let removedColumns: [String]

    /// Creates a new instance.
    /// - Parameters:
    ///   - v1: The v1.
    ///   - v2: The v2.
    public init(v1: DataFrame, v2: DataFrame) {
        self.rowDelta = v2.rowCount - v1.rowCount
        let v1Cols = Set(v1.columnNames)
        let v2Cols = Set(v2.columnNames)
        self.columnDelta = v2Cols.count - v1Cols.count
        self.addedColumns = Array(v2Cols.subtracting(v1Cols)).sorted()
        self.removedColumns = Array(v1Cols.subtracting(v2Cols)).sorted()
    }

    /// Summary.
    /// - Returns: A `String` result.
    public func summary() -> String {
        return "Row delta: \(rowDelta), Column delta: \(columnDelta), Added: \(addedColumns), Removed: \(removedColumns)"
    }
}
