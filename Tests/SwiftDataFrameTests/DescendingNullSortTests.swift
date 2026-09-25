import Testing
import SwiftDataFrame

@Suite("Descending sort with missing values")
struct DescendingNullSortTests {
    @Test func descendingKeepsNullLast() {
        let column = TypedColumn<Double>(name: "reading", values: [2, nil, 1])
        let indices = column.sortedIndices(ascending: false)
        #expect(indices == [0, 2, 1])
    }
}

@Suite("Descending sort callers")
struct DescendingNullSortCallerTests {
    @Test func stringAndBoolKeepNullLast() {
        let strings = TypedColumn<String>(name: "label", values: ["b", nil, "a"])
        let bools = TypedColumn<Bool>(name: "enabled", values: [true, nil, false])
        #expect(strings.sortedIndices(ascending: false) == [0, 2, 1])
        #expect(bools.sortedIndices(ascending: false) == [0, 2, 1])
    }

    @Test func dataFrameSortPreservesRowAssociation() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Double>(name: "reading", values: [2, nil, 1]),
            TypedColumn<String>(name: "row", values: ["two", "missing", "one"])
        ])
        let sorted = try frame.sortBy("reading", ascending: false)
        #expect(sorted[column: "reading", as: Double.self]?.values == [2, 1, nil])
        #expect(sorted[column: "row", as: String.self]?.values == ["two", "one", "missing"])
    }
}
