import Foundation

/// Defines the strategy for joining two DataFrames.
public enum JoinKind: Sendable {
    /// Returns only rows with matching keys in both DataFrames.
    case inner
    /// Returns all rows from left DataFrame, and matching rows from right.
    case left
    /// Returns all rows from right DataFrame, and matching rows from left.
    case right
    /// Returns all rows when there is a match in either left or right DataFrame.
    case outer
}

extension DataFrame {
    /// Joins two DataFrames on a common key column using an efficient hash join algorithm.
    ///
    /// - Parameters:
    ///   - other: The DataFrame to join with `self`.
    ///   - key: The column name to join on (must exist in both DataFrames).
    ///   - how: The type of join (`.inner`, `.left`, `.right`, `.outer`). Default is `.inner`.
    /// - Returns: A new DataFrame containing the joined columns and matching rows.
    /// - Throws: `SwiftMLError.columnNotFound` if key column is missing in either DataFrame.
    public func join(
        _ other: DataFrame,
        on key: String,
        how: JoinKind = .inner
    ) throws -> DataFrame {
        try joinSIMD(other, on: key, how: how)
    }
}
