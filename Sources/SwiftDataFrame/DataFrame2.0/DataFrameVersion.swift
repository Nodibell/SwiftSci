import Foundation

/// Snapshot metadata for DataFrame versioning.
public struct DataFrameVersionSnapshot: Sendable, Codable, Equatable {
    public let version: Int
    public let tag: String
    public let timestamp: Date
    public let rowCount: Int
    public let columnCount: Int

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
    public let rowDelta: Int
    public let columnDelta: Int
    public let addedColumns: [String]
    public let removedColumns: [String]

    public init(v1: DataFrame, v2: DataFrame) {
        self.rowDelta = v2.rowCount - v1.rowCount
        let v1Cols = Set(v1.columnNames)
        let v2Cols = Set(v2.columnNames)
        self.columnDelta = v2Cols.count - v1Cols.count
        self.addedColumns = Array(v2Cols.subtracting(v1Cols)).sorted()
        self.removedColumns = Array(v1Cols.subtracting(v2Cols)).sorted()
    }

    public func summary() -> String {
        return "Row delta: \(rowDelta), Column delta: \(columnDelta), Added: \(addedColumns), Removed: \(removedColumns)"
    }
}
