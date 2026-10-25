import Testing
@testable import SwiftDataFrame

@Suite("Empty joins preserve relational schema")
struct EmptyJoinSchemaTests {
    private func frame(_ keys: [Double], _ labels: [String]) throws -> DataFrame {
        let compact = CompactNumericColumn(keys)
        return try DataFrame(columns: [
            TypedColumn<Double>(name: "key", compactColumn: compact),
            TypedColumn<String>(name: "label", values: labels)
        ])
    }

    @Test func disjointInnerJoinRetainsTypedEmptyColumns() throws {
        let left = try frame([1], ["left"])
        let right = try frame([2], ["right"])
        let joined = try left.join(right, on: "key")
        try checkEmpty(joined)
        #expect(joined.columnNames == ["key", "label_x", "label_y"])
    }

    @Test(arguments: [JoinKind.inner, .left, .right, .outer])
    func emptyInputsRetainSchema(kind: JoinKind) throws {
        let full = try frame([1], ["present"])
        let empty = full.gathered(at: [])
        for (left, right) in [(empty, full), (full, empty), (empty, empty)] {
            let joined = try left.joinSIMD(right, on: "key", how: kind)
            #expect(joined.columnNames == ["key", "label_x", "label_y"])
            #expect(joined.columns.map(\.dtype) == [.float64, .utf8, .utf8])
            let expectedRows: Int
            switch kind {
            case .inner: expectedRows = 0
            case .left: expectedRows = left.rowCount
            case .right: expectedRows = right.rowCount
            case .outer: expectedRows = left.rowCount + right.rowCount
            }
            #expect(joined.rowCount == expectedRows)
            if expectedRows == 0 { try checkEmpty(joined) }
        }
    }

    @Test func invalidKeysStillFailBeforeEmptyAssembly() throws {
        let empty = try frame([], [])
        #expect(throws: SwiftMLError.self) { try empty.join(empty, on: "missing") }
        let incompatible = try DataFrame(columns: [TypedColumn<Int64>(name: "key", values: [Int64]())])
        #expect(throws: SwiftMLError.self) { try empty.join(incompatible, on: "key") }
    }

    private func checkEmpty(_ frame: DataFrame) throws {
        #expect(frame.rowCount == 0)
        let key = try #require(frame[column: "key", as: Double.self])
        #expect(key.compactNumericStorage != nil)
        #expect(frame.columns.allSatisfy { $0.count == 0 && $0.nullCount == 0 })
        let matrix = try frame.prepareNumericMatrix(["key"])
        #expect(matrix.rowCount == 0 && matrix.columnCount == 1 && matrix.values.isEmpty)
    }
}
