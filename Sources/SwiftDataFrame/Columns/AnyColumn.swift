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

    /// Returns a new column shifted by `by` rows, filled with nulls at the edges.
    func lagged(by offset: Int) -> any AnyColumn
}
