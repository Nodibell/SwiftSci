import Foundation
import Accelerate

/// Single-column filters that dispatch to typed comparison loops when supported.
/// The filter selects row indices, then gathers the result columns.
/// Selection is linear in the input row count; gathering depends on the number
/// of selected rows and output columns.
public extension DataFrame {

    /// Filters rows using a single-column condition.
    /// Numeric columns compare in typed loops without per-row type erasure.
    /// Unsupported conditions retain the generic comparison fallback.
    ///
    /// - Parameters:
    ///   - name:      Name of the column to filter on.
    ///   - condition: A `FilterCondition` describing the comparison operator and threshold.
    /// - Returns: A new `DataFrame` containing only the rows that satisfy the condition.
    /// - Throws: `DataFrameError.columnNotFound` when the column does not exist.
    func filterFast(column name: String, where condition: FilterCondition) throws -> DataFrame {
        guard let col = _columns[name] else {
            throw SwiftMLError.columnNotFound(name)
        }

        // Resolve the column type before evaluating rows.
        if let indices = col.filteredIndices(matching: condition) {
            return gathered(at: indices)
        }

        // Fallback: scalar mask path (Bool columns, unsupported operators).
        var passingIndices = [Int]()
        passingIndices.reserveCapacity(shape.rows / 2)
        for i in 0..<shape.rows {
            if condition.evaluate(value: col.value(at: i)) {
                passingIndices.append(i)
            }
        }
        return gathered(at: passingIndices)

    }

    /// Filters rows using a Double threshold.
    ///
    /// Sugar overload for the common case of filtering a numeric column with a scalar
    /// threshold, avoiding the need to construct a `FilterCondition` manually.
    ///
    /// ```swift
    /// let highScores = try df.filterFast(column: "score", op: .greaterThanOrEqual, threshold: 85.0)
    /// ```
    ///
    /// - Parameters:
    ///   - name:      Name of the column to filter on.
    ///   - op:        The comparison operator.
    ///   - threshold: The Double scalar threshold.
    /// - Returns: A new `DataFrame` containing only the rows that satisfy `column op threshold`.
    /// - Throws: `DataFrameError.columnNotFound` when the column does not exist.
    func filterFast(column name: String, op: FilterOp, threshold: Double) throws -> DataFrame {
        let condition: FilterCondition
        switch op {
        case .greaterThan:          condition = .greaterThan(threshold)
        case .greaterThanOrEqual:   condition = .greaterThanOrEqual(threshold)
        case .lessThan:             condition = .lessThan(threshold)
        case .lessThanOrEqual:      condition = .lessThanOrEqual(threshold)
        case .equals:               condition = .equals(threshold)
        case .notEquals:            condition = .notEquals(threshold)
        }
        return try filterFast(column: name, where: condition)
    }
}

/// Comparison operator used by the `filterFast(column:op:threshold:)` sugar overload.
public enum FilterOp: Sendable {
    case greaterThan
    case greaterThanOrEqual
    case lessThan
    case lessThanOrEqual
    case equals
    case notEquals
}
