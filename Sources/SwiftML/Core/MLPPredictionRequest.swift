import SwiftDataFrame

/// A source snapshot, logical occurrence selection, and deferred sparse edits.
/// Request arrays already exist outside preparation admission. Gathering and ordering
/// occur only during preparation. A small selection still retains the complete source.
public struct MLPPredictionRequest: Sendable {
    let source: PreparedNumericBatch
    let selectedRows: [Int]?
    private(set) var edits: [ColumnEdits] = []
    struct ColumnEdits: Sendable { let column: Int; var updates: [Update] }
    struct Update: Sendable { let rows: [Int]; let values: [Double?] }

    /// Retains all source rows without creating an identity occurrence array.
    public init(source: PreparedNumericBatch) { self.source = source; selectedRows = nil }

    /// Retains source positions in logical order, including duplicate occurrences.
    /// Out-of-bounds positions throw before the request changes.
    public init(source: PreparedNumericBatch, selectingRows rows: [Int]) throws {
        guard rows.allSatisfy({ $0 >= 0 && $0 < source.rowCount }) else {
            throw SwiftMLError.invalidInput("Prediction selection is out of bounds")
        }
        self.source = source; selectedRows = rows
    }

    /// Number of selected logical occurrences.
    public var rowCount: Int { selectedRows?.count ?? source.rowCount }

    /// Records edits in selected logical-row coordinates. Repeated writes use the last value.
    /// Nil marks missing data; the fitted plan supplies its imputation value. Invalid input
    /// throws before changing the request. Shared request values remain snapshots.
    public mutating func updateColumn(at column: Int, rows: [Int], values: [Double?]) throws {
        guard column >= 0, column < source.columnCount else {
            throw SwiftMLError.invalidInput("Prediction edit column is out of bounds")
        }
        guard rows.count == values.count else {
            throw SwiftMLError.dimensionMismatch(expected: rows.count, got: values.count)
        }
        guard rows.allSatisfy({ $0 >= 0 && $0 < rowCount }) else {
            throw SwiftMLError.invalidInput("Prediction edit occurrence is out of bounds")
        }
        guard !rows.isEmpty else { return }
        let update = Update(rows: rows, values: values)
        if let index = edits.firstIndex(where: { $0.column == column }) {
            edits[index].updates.append(update)
        } else { edits.append(ColumnEdits(column: column, updates: [update])) }
    }

    func materialized() throws -> IndexedNumericBatch {
        var view = try IndexedNumericBatch(source, selectingRows: selectedRows ?? Array(0..<source.rowCount))
        for column in edits {
            for update in column.updates {
                try Task.checkCancellation()
                try view.updateColumn(at: column.column, rows: update.rows, values: update.values)
            }
        }
        return view
    }
}
