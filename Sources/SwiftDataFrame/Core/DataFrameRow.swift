/// A lightweight view of a single row in a DataFrame.
/// Used as the argument type for predicate-based filtering.
///
/// Values are resolved lazily on access so predicates that touch only one
/// column (the common case) do not materialize the entire row.
public struct DataFrameRow: @unchecked Sendable {

    /// Ordered column names for this row.
    public let columnNames: [String]

    /// The index.
    public internal(set) var index: Int
    private let columnMap: [String: any AnyColumn]

    init(columnNames: [String], index: Int, columnMap: [String: any AnyColumn]) {
        self.columnNames = columnNames
        self.index = index
        self.columnMap = columnMap
    }

    /// Returns the value for `column` as the given type, or nil if null / wrong type.
    public func value<T: SupportedType>(column: String, as type: T.Type = T.self) -> T? {
        columnMap[column]?.value(at: index) as? T
    }

    /// Returns the raw value as `Any?`.
    public subscript(column: String) -> Any? {
        columnMap[column]?.value(at: index)
    }

    /// Returns the typed value for `column`, or nil if null / wrong type.
    public subscript<T: SupportedType>(_ name: String, as type: T.Type) -> T? {
        guard let col = columnMap[name] as? TypedColumn<T> else { return nil }
        return col[index]
    }

    /// Convenience typed accessor for Double columns.
    public func double(_ name: String) -> Double? { self[name, as: Double.self] }

    /// Convenience typed accessor for String columns.
    public func string(_ name: String) -> String? { self[name, as: String.self] }

    /// Convenience typed accessor for Int64 columns.
    public func int(_ name: String) -> Int64?     { self[name, as: Int64.self] }

    /// Whether the value for `column` is null.
    public func isNull(column: String) -> Bool {
        guard let col = columnMap[column] else { return true }
        return col.value(at: index) == nil
    }
}

/// A zero-allocation lazy sequence for iterating over DataFrame rows.
public struct DataFrameRowSequence: Sequence, @unchecked Sendable {
    /// Number of rows in the sequence.
    public let count: Int
    private let columnNames: [String]
    private let columnMap: [String: any AnyColumn]

    init(count: Int, columnNames: [String], columnMap: [String: any AnyColumn]) {
        self.count = count
        self.columnNames = columnNames
        self.columnMap = columnMap
    }

    /// Creates an iterator for row iteration.
    public func makeIterator() -> Iterator {
        Iterator(count: count, row: DataFrameRow(columnNames: columnNames, index: 0, columnMap: columnMap))
    }

    /// Iterator over DataFrame rows reusing an internal row view.
    public struct Iterator: IteratorProtocol {
        private let count: Int
        private var row: DataFrameRow
        private var currentIndex: Int = 0

        init(count: Int, row: DataFrameRow) {
            self.count = count
            self.row = row
        }

        /// Advances to the next row.
        public mutating func next() -> DataFrameRow? {
            guard currentIndex < count else { return nil }
            row.index = currentIndex
            currentIndex += 1
            return row
        }
    }
}

