import Foundation

/// A tabular data structure with named, typed columns.
/// Columns are stored in-order; all columns must have the same `count`.
/// `DataFrame` is a value type — all mutations return new instances.
public struct DataFrame: Sendable {

    // MARK: – Internal storage

    /// Ordered columns (order preserved from initialisation).
    internal let _columns: [String: any AnyColumn]
    internal let _columnOrder: [String]

    // MARK: – Static helpers

    public static let empty = DataFrame(_columns: [:], _columnOrder: [])

    private init(_columns: [String: any AnyColumn], _columnOrder: [String]) {
        self._columns      = _columns
        self._columnOrder  = _columnOrder
    }

    // MARK: – Initialisation

    /// Creates a DataFrame from an ordered list of columns.
    /// - Throws: `DataFrameError.emptySchema` if `columns` is empty.
    ///           `DataFrameError.columnLengthMismatch` if columns differ in length.
    ///           `DataFrameError.duplicateColumnName` if two columns share a name.
    public init(columns: [any AnyColumn]) throws {
        guard !columns.isEmpty else { throw DataFrameError.emptySchema }

        let expectedCount = columns[0].count
        var colMap: [String: any AnyColumn] = [:]
        var colOrder: [String] = []

        for col in columns {
            guard col.count == expectedCount else {
                throw DataFrameError.columnLengthMismatch(
                    expected: expectedCount, got: col.count, column: col.name
                )
            }
            guard colMap[col.name] == nil else {
                throw DataFrameError.duplicateColumnName(col.name)
            }
            colMap[col.name] = col
            colOrder.append(col.name)
        }

        self._columns     = colMap
        self._columnOrder = colOrder
    }

    /// Creates a DataFrame by reading a CSV file.
    public init(csv url: URL, options: CSVReadOptions = .default) async throws {
        let df = try await CSVReader.read(url: url, options: options)
        self = df
    }

    /// Creates a DataFrame by reading a JSON file (array of objects).
    public init(json url: URL) async throws {
        let df = try await JSONReader.read(url: url)
        self = df
    }

    // MARK: – Metadata

    public var shape: (rows: Int, columns: Int) {
        (rows: _columns.first.map { $0.value.count } ?? 0,
         columns: _columnOrder.count)
    }

    public var columnNames: [String] { _columnOrder }

    public var dtypes: [String: ColumnDType] {
        _columns.mapValues { $0.dtype }
    }

    /// All columns in their original order.
    public var columns: [any AnyColumn] {
        _columnOrder.compactMap { _columns[$0] }
    }

    public func schema() -> Schema {
        Schema(fields: columns.map { col in
            Schema.Field(name: col.name, dtype: col.dtype, nullable: col.nullCount > 0)
        })
    }

    // MARK: – Subscript access

    public subscript(column name: String) -> (any AnyColumn)? {
        _columns[name]
    }

    public subscript<T: SupportedType>(column name: String, as type: T.Type) -> TypedColumn<T>? {
        _columns[name] as? TypedColumn<T>
    }

    /// Returns the row at `index` as a dictionary.
    public func row(at index: Int) -> [String: Any?] {
        guard index >= 0 && index < shape.rows else { return [:] }
        var result: [String: Any?] = [:]
        for col in columns { result[col.name] = col.value(at: index) }
        return result
    }

    // MARK: – Selection

    /// Returns a new DataFrame with only the specified columns.
    public func select(_ names: String...) throws -> DataFrame {
        try selectArray(names)
    }

    public func select(_ names: [String]) throws -> DataFrame {
        try selectArray(names)
    }

    private func selectArray(_ names: [String]) throws -> DataFrame {
        var cols: [any AnyColumn] = []
        for name in names {
            guard let col = _columns[name] else {
                throw DataFrameError.columnNotFound(name)
            }
            cols.append(col)
        }
        return try DataFrame(columns: cols)
    }

    /// Returns a new DataFrame without the specified columns.
    public func drop(_ names: String...) throws -> DataFrame {
        try dropArray(names)
    }

    public func drop(_ names: [String]) throws -> DataFrame {
        try dropArray(names)
    }

    private func dropArray(_ names: [String]) throws -> DataFrame {
        let nameSet = Set(names)
        for n in names {
            guard _columns[n] != nil else { throw DataFrameError.columnNotFound(n) }
        }
        let remaining = _columnOrder.filter { !nameSet.contains($0) }
        let cols = remaining.compactMap { _columns[$0] }
        guard !cols.isEmpty else { return DataFrame.empty }
        return try DataFrame(columns: cols)
    }

    /// Returns the first `n` rows.
    public func head(_ n: Int = 5) -> DataFrame {
        slice(from: 0, count: n)
    }

    /// Returns the last `n` rows.
    public func tail(_ n: Int = 5) -> DataFrame {
        let total = shape.rows
        let from  = Swift.max(0, total - n)
        return slice(from: from, count: total - from)
    }

    /// Returns a random sample of `n` rows.
    public func sample(n: Int, seed: UInt64? = nil) -> DataFrame {
        let total = shape.rows
        guard n > 0 && total > 0 else { return DataFrame.empty }
        let actualN = Swift.min(n, total)

        var rng = SeededRNG(seed: seed ?? UInt64(Date().timeIntervalSince1970 * 1000))
        var indices = Array(0..<total)
        // Fisher-Yates shuffle first `actualN` elements
        for i in 0..<actualN {
            let j = i + Int(rng.next() % UInt64(total - i))
            indices.swapAt(i, j)
        }
        let selected = Array(indices.prefix(actualN)).sorted()
        return rows(at: selected)
    }

    // MARK: – Filtering

    /// Filters rows using a predicate closure.
    ///
    /// The row view resolves column values lazily, so predicates that only
    /// read one or two columns avoid materializing the full row.
    public func filter(_ predicate: (DataFrameRow) -> Bool) -> DataFrame {
        guard shape.rows > 0 else { return DataFrame.empty }
        let map = _columns
        let names = columnNames
        var mask = [Bool](repeating: false, count: shape.rows)
        for i in 0..<shape.rows {
            let row = DataFrameRow(columnNames: names, index: i, columnMap: map)
            mask[i] = predicate(row)
        }
        return applyMask(mask)
    }

    /// Filters rows by a condition on a single column.
    public func filter(column name: String, where condition: FilterCondition) throws -> DataFrame {
        guard let col = _columns[name] else {
            throw DataFrameError.columnNotFound(name)
        }

        // Numeric vectorized fast path for Double columns (common benchmark / pandas case).
        if let typed = col as? TypedColumn<Double>,
           let mask = typed.mask(matching: condition) {
            return applyMask(mask)
        }

        var mask = [Bool](repeating: false, count: shape.rows)
        for i in 0..<shape.rows {
            mask[i] = condition.evaluate(value: col.value(at: i))
        }
        return applyMask(mask)
    }

    // MARK: – Transformation

    /// Returns a new DataFrame with the column replaced or added.
    public func withColumn(_ name: String, column: any AnyColumn) throws -> DataFrame {
        let expectedRows = _columnOrder.isEmpty ? column.count : shape.rows
        guard column.count == expectedRows else {
            throw DataFrameError.columnLengthMismatch(
                expected: expectedRows, got: column.count, column: name
            )
        }
        let newCol = column.renamed(to: name)
        var newMap   = _columns
        var newOrder = _columnOrder

        newMap[name] = newCol
        if !newOrder.contains(name) { newOrder.append(name) }

        return DataFrame(_columns: newMap, _columnOrder: newOrder)
    }

    /// Returns a new DataFrame with a lagged column added.
    public func withLaggedColumn(column name: String, by offset: Int, newName: String) throws -> DataFrame {
        guard let col = _columns[name] else { throw DataFrameError.columnNotFound(name) }
        let laggedCol = col.lagged(by: offset).renamed(to: newName)
        return try withColumn(newName, column: laggedCol)
    }

    /// Returns a new DataFrame with a column renamed.
    public func renameColumn(_ old: String, to new: String) throws -> DataFrame {
        guard let col = _columns[old] else { throw DataFrameError.columnNotFound(old) }
        var newMap   = _columns
        var newOrder = _columnOrder

        newMap.removeValue(forKey: old)
        newMap[new] = col.renamed(to: new)
        if let idx = newOrder.firstIndex(of: old) { newOrder[idx] = new }

        return DataFrame(_columns: newMap, _columnOrder: newOrder)
    }

    /// Casts a column to a new type.
    public func castColumn<T: SupportedType>(_ name: String, to type: T.Type) throws -> DataFrame {
        guard let col = _columns[name] else { throw DataFrameError.columnNotFound(name) }

        // Try to re-parse each value via SupportedType.parse(from:)
        let newValues: [T?] = (0..<col.count).map { i in
            guard let v = col.value(at: i) else { return nil }
            let str = "\(v)"
            return T.parse(from: str)
        }

        // Validate at least one succeeded if source had non-null values
        let sourceNonNull = col.count - col.nullCount
        let castNonNull   = newValues.filter { $0 != nil }.count
        if sourceNonNull > 0 && castNonNull == 0 {
            throw DataFrameError.castFailed(column: name, targetType: "\(T.self)")
        }

        let newCol = TypedColumn<T>(name: name, values: newValues)
        return try withColumn(name, column: newCol)
    }

    /// Returns a new DataFrame sorted by the given column.
    public func sortBy(_ column: String, ascending: Bool = true) throws -> DataFrame {
        guard let col = _columns[column] else { throw DataFrameError.columnNotFound(column) }
        let indices = col.sortedIndices(ascending: ascending)
        return rows(at: indices)
    }

    /// Returns a `GroupedDataFrame` for aggregation.
    public func groupBy(_ columns: String...) -> GroupedDataFrame {
        GroupedDataFrame(dataFrame: self, groupColumns: columns)
    }

    // MARK: – I/O

    /// Writes the DataFrame to a CSV file.
    public func writeCSV(to url: URL) async throws {
        try await CSVWriter.write(self, to: url)
    }

    /// Prints a formatted table to stdout.
    public func debugPrint(maxRows: Int = 20) {
        let names  = columnNames
        let rows   = Swift.min(shape.rows, maxRows)

        // Column widths
        var widths = names.map { $0.count }
        for i in 0..<rows {
            for (ci, col) in columns.enumerated() {
                let s = col.value(at: i).map { "\($0)" } ?? "null"
                widths[ci] = Swift.max(widths[ci], s.count)
            }
        }

        func pad(_ s: String, _ w: Int) -> String { s + String(repeating: " ", count: w - s.count) }
        let header = names.enumerated().map { pad($0.element, widths[$0.offset]) }.joined(separator: " | ")
        let sep    = widths.map { String(repeating: "-", count: $0) }.joined(separator: "-+-")

        print(header)
        print(sep)
        for i in 0..<rows {
            let row = columns.enumerated().map { (ci, col) in
                pad(col.value(at: i).map { "\($0)" } ?? "null", widths[ci])
            }.joined(separator: " | ")
            print(row)
        }
        if shape.rows > maxRows {
            print("... (\(shape.rows - maxRows) more rows)")
        }
        print("\n[\(shape.rows) rows × \(shape.columns) columns]")
    }

    // MARK: – Private helpers

    private func slice(from start: Int, count n: Int) -> DataFrame {
        let total  = shape.rows
        guard total > 0, n > 0 else { return DataFrame.empty }
        let upper  = Swift.min(start + n, total)
        let indices = Array(start..<upper)
        return rows(at: indices)
    }

    private func rows(at indices: [Int]) -> DataFrame {
        guard !indices.isEmpty else { return DataFrame.empty }
        let newCols: [any AnyColumn] = columns.map { $0.gathered(at: indices) }
        return (try? DataFrame(columns: newCols)) ?? DataFrame.empty
    }

    private func applyMask(_ mask: [Bool]) -> DataFrame {
        var indices: [Int] = []
        indices.reserveCapacity(mask.count / 2)
        for (i, keep) in mask.enumerated() where keep {
            indices.append(i)
        }
        guard !indices.isEmpty else { return DataFrame.empty }
        return rows(at: indices)
    }
}

// MARK: – Factory helper

/// Reconstruct a typed AnyColumn from raw Any? values.
internal func makeColumn(name: String, dtype: ColumnDType, rawValues: [Any?]) -> any AnyColumn {
    switch dtype {
    case .int64:
        return TypedColumn<Int64>(name: name, values: rawValues.map { $0 as? Int64 })
    case .int32:
        return TypedColumn<Int32>(name: name, values: rawValues.map { $0 as? Int32 })
    case .float64:
        return TypedColumn<Double>(name: name, values: rawValues.map { $0 as? Double })
    case .float32:
        return TypedColumn<Float>(name: name, values: rawValues.map { $0 as? Float })
    case .boolean:
        return TypedColumn<Bool>(name: name, values: rawValues.map { $0 as? Bool })
    case .date32:
        return TypedColumn<Date>(name: name, values: rawValues.map { $0 as? Date })
    case .utf8:
        return TypedColumn<String>(name: name,
            values: rawValues.map { v in v.map { "\($0)" } })
    }
}

// MARK: – Seeded RNG (for reproducible sample)

private struct SeededRNG {
    private var state: UInt64

    init(seed: UInt64) { self.state = seed == 0 ? 1 : seed }

    mutating func next() -> UInt64 {
        // xorshift64
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}
