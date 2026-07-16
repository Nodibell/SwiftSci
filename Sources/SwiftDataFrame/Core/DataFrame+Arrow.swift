import Foundation
import Arrow

extension DataFrame {
    
    /// Initialises a `DataFrame` from an Apache Arrow `ArrowTable`.
    public init(arrowTable: ArrowTable) throws {
        let df = try ArrowTableBridge.toDataFrame(arrowTable)
        try self.init(columns: df.columns)
    }
    
    /// Converts the `DataFrame` into an Apache Arrow `ArrowTable`.
    public func toArrowTable() throws -> ArrowTable {
        try ArrowTableBridge.toArrowTable(self)
    }
}
