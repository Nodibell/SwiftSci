/// Source snapshot with selected occurrences and native storage only for edited columns.
package struct IndexedNumericBatch: Sendable {
    package let source: PreparedNumericBatch
    package let sourcePositions: [Int]
    package private(set) var editedColumns: [IndexedEditedColumn] = []
    package var rowCount: Int { sourcePositions.count }
    package var columnCount: Int { source.columnCount }
    package var columnNames: [String] { source.columnNames }
    package var originalRowIndices: [Int] { sourcePositions.map { source.originalRowIndex(at: $0) } }

    package init(_ source: PreparedNumericBatch, selectingRows positions: [Int]) throws {
        guard positions.allSatisfy({ $0 >= 0 && $0 < source.rowCount }) else {
            throw SwiftMLError.invalidParameter("Indexed selection is out of bounds")
        }
        let (_, overflow) = positions.count.multipliedReportingOverflow(by: source.columnCount)
        guard !overflow else { throw SwiftMLError.invalidParameter("Indexed element count overflow") }
        self.source = source
        sourcePositions = positions
    }

    package subscript(row: Int, column: Int) -> Double? {
        if let edited = editedColumns.first(where: { $0.index == column }) { return edited.data[row] }
        return source[sourcePositions[row], column]
    }

    package mutating func updateColumn(at column: Int, rows: [Int], values: [Double?]) throws {
        guard column >= 0, column < columnCount else {
            throw SwiftMLError.invalidParameter("Indexed update column is out of bounds")
        }
        guard rows.count == values.count else {
            throw SwiftMLError.dimensionMismatch(expected: rows.count, got: values.count)
        }
        guard rows.allSatisfy({ $0 >= 0 && $0 < rowCount }) else {
            throw SwiftMLError.invalidParameter("Indexed update position is out of bounds")
        }
        guard !rows.isEmpty else { return }
        if let position = editedColumns.firstIndex(where: { $0.index == column }) {
            editedColumns[position].data.update(at: rows, with: values)
        } else {
            var data = source.columns[column].gathered(at: sourcePositions)
            data.update(at: rows, with: values)
            let position = editedColumns.firstIndex(where: { $0.index > column }) ?? editedColumns.endIndex
            editedColumns.insert(IndexedEditedColumn(index: column, data: data), at: position)
        }
    }
    /// Orders independent rows by physical position in the captured source.
    package func sourceOrdered(prefixCount: Int? = nil) throws -> (batch: IndexedNumericBatch, logicalPositions: [Int]?) {
        let prefix = prefixCount ?? rowCount
        guard prefix >= 0, prefix <= rowCount else {
            throw SwiftMLError.invalidParameter("Source-order prefix is out of bounds")
        }
        var ordered = true
        if prefix > 1 {
            for r in 1..<prefix where sourcePositions[r - 1] > sourcePositions[r] { ordered = false; break }
        }
        guard !ordered else { return (self, nil) }
        let logicalPositions = (0..<prefix).sorted {
            let left = sourcePositions[$0], right = sourcePositions[$1]
            return left == right ? $0 < $1 : left < right
        } + Array(prefix..<rowCount)
        var result = try IndexedNumericBatch(source,
            selectingRows: logicalPositions.map { sourcePositions[$0] })
        result.editedColumns = editedColumns.map {
            IndexedEditedColumn(index: $0.index, data: $0.data.gathered(at: logicalPositions))
        }
        return (result, logicalPositions)
    }

}

package struct IndexedEditedColumn: Sendable {
    package let index: Int
    package var data: CompactNumericColumn
}
