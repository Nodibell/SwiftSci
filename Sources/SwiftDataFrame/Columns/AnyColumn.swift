/// Type-erased column that can be stored in a DataFrame.
/// All concrete column types must conform to this protocol.
public protocol AnyColumn: Sendable {
    /// The column name.
    var name: String { get }
    /// The element data type.
    var dtype: ColumnDType { get }
    /// Total number of elements (including nulls).
    var count: Int { get }
    /// Number of null (missing) elements.
    var nullCount: Int { get }

    /// Returns a new column keeping only elements where mask is true.
    /// - Parameter mask: Boolean mask of the same length as `count`.
    /// - Throws: `DataFrameError.columnLengthMismatch` if mask length differs.
    func filtered(by mask: [Bool]) throws -> any AnyColumn

    /// Returns a new column with values gathered in `indices` order.
    func gathered(at indices: [Int]) -> any AnyColumn

    /// Returns a new column with values gathered at `indices` where nil indices yield null elements.
    func gatheredWithNulls(at indices: [Int?], newName: String?) -> any AnyColumn

    /// Fast path for joining key columns: gathers from self where leftIndices is non-nil, or rightColumn where rightIndices is non-nil.
    func gatheredJoinKey(leftIndices: [Int?], rightColumn: any AnyColumn, rightIndices: [Int?], name: String) -> any AnyColumn

    /// Returns the element at the given index as `Any?` (nil = null).
    func value(at index: Int) -> Any?

    /// Returns all non-null elements as `[Double]`. Returns nil if not numeric.
    func toDoubles() -> [Double]?

    /// Returns all elements as `[String]` (nulls become "null").
    func toStrings() -> [String]

    /// Returns a renamed copy of this column.
    func renamed(to newName: String) -> any AnyColumn
    /// Returns indices sorted by value (nulls last).
    func sortedIndices(ascending: Bool) -> [Int]

    /// Returns a new column containing only unique elements (preserving order of first appearance).
    var unique: any AnyColumn { get }

    /// Returns a new column shifted by `by` rows, filled with nulls at the edges.
    func lagged(by offset: Int) -> any AnyColumn

    /// Fast path: returns array of row indices matching the given condition directly without building a [Bool] mask.
    func filteredIndices(matching condition: FilterCondition) -> [Int]?
}

extension AnyColumn {
    /// Filtered indices.
    /// - Returns: A `[Int]?` result.
    /// - Parameter condition: Filter condition to match.
    public func filteredIndices(matching condition: FilterCondition) -> [Int]? { nil }

    /// Returns a new column with values gathered at `indices` where nil indices yield null elements.
    public func gatheredWithNulls(at indices: [Int?], newName: String? = nil) -> any AnyColumn {
        let n = indices.count
        var rawVals = [Any?]()
        rawVals.reserveCapacity(n)
        for idx in indices {
            if let idx = idx {
                rawVals.append(self.value(at: idx))
            } else {
                rawVals.append(nil)
            }
        }
        return makeColumn(name: newName ?? self.name, dtype: self.dtype, rawValues: rawVals)
    }

    /// Fast path for joining key columns.
    public func gatheredJoinKey(leftIndices: [Int?], rightColumn: any AnyColumn, rightIndices: [Int?], name: String) -> any AnyColumn {
        let total = leftIndices.count
        var rawVals = [Any?]()
        rawVals.reserveCapacity(total)
        for i in 0..<total {
            if let lIdx = leftIndices[i] {
                rawVals.append(self.value(at: lIdx))
            } else if let rIdx = rightIndices[i] {
                rawVals.append(rightColumn.value(at: rIdx))
            } else {
                rawVals.append(nil)
            }
        }
        return makeColumn(name: name, dtype: self.dtype, rawValues: rawVals)
    }
}

