import Foundation
import Arrow

/// Strategy for handling null values and validity bitmaps when converting Apache Arrow tables to DataFrame columns.
public enum ArrowNullStrategy: Sendable {
    /// Preserves null values as `nil` (representing missing values in `TypedColumn`).
    case preserve

    /// Replaces missing float values (`Float32`, `Float64`) with IEEE 754 `Float.nan` / `Double.nan`.
    case nan

    /// Replaces missing numeric values with zero (`0` or `0.0`).
    case zero
}

extension DataFrame {
    
    /// Initialises a `DataFrame` from an Apache Arrow `ArrowTable`.
    ///
    /// - Parameters:
    ///   - arrowTable: The Apache Arrow table to import.
    ///   - nullStrategy: Strategy to apply to null entries indicated by Arrow validity bitmaps. Defaults to `.preserve`.
    /// - Throws: ``SwiftMLError`` or Arrow bridging errors if conversion fails.
    public init(arrowTable: ArrowTable, nullStrategy: ArrowNullStrategy = .preserve) throws {
        let df = try ArrowTableBridge.toDataFrame(arrowTable, nullStrategy: nullStrategy)
        try self.init(columns: df.columns)
    }
    
    /// Converts the `DataFrame` into an Apache Arrow `ArrowTable`.
    public func toArrowTable() throws -> ArrowTable {
        try ArrowTableBridge.toArrowTable(self)
    }
}
