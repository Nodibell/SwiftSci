import Foundation
import Testing
import SwiftDataFrame

@Suite("Column null counts")
struct ColumnNullCountTests {
    private func checkColumn<T: SupportedType>(_ values: [T?]) throws {
        let column = TypedColumn<T>(name: "value", values: values)
        let erased: any AnyColumn = column
        #expect(column.nullCount == 2)

        let indices = [3, 1, 0, 1, 2]
        let gathered = try #require(erased.gathered(at: indices) as? TypedColumn<T>)
        #expect(gathered.values == indices.map { values[$0] })
        #expect(gathered.nullCount == 3)
        #expect(gathered.name == column.name)
        #expect(gathered.dtype == column.dtype)

        let empty = try #require(erased.gathered(at: []) as? TypedColumn<T>)
        #expect(empty.count == 0)
        #expect(empty.nullCount == 0)

        let filtered = try #require(erased.filtered(by: [true, false, true, true]) as? TypedColumn<T>)
        #expect(filtered.values == [values[0], values[2], values[3]])
        #expect(filtered.nullCount == 1)
        #expect(column.values == values)
        #expect(column.nullCount == 2)
    }

    @Test("Construction and row selection preserve null counts across column types")
    func typedColumnSelection() throws {
        try checkColumn([Double(1), nil, 3, nil])
        try checkColumn([Float(1), nil, 3, nil])
        try checkColumn([Int64(1), nil, 3, nil])
        try checkColumn([Int32(1), nil, 3, nil])
        try checkColumn(["one", nil, "three", nil])
        try checkColumn([true, nil, false, nil])
        try checkColumn([Date(timeIntervalSince1970: 1), nil, Date(timeIntervalSince1970: 3), nil])
    }

    @Test("Empty, all-null and non-null columns report exact counts")
    func boundaryCounts() {
        #expect(TypedColumn<Int64>(name: "empty", values: [Int64?]()).nullCount == 0)
        #expect(TypedColumn<String>(name: "nulls", values: [nil, nil, nil]).nullCount == 3)
        #expect(TypedColumn<Int64>(name: "valid", values: [1, 2, 3]).nullCount == 0)
        #expect(TypedColumn<Double>(name: "special", values: [.nan, .infinity, -.infinity, nil]).nullCount == 1)
    }

    @Test("Column filtering preserves companion column values, order and null counts")
    func dataframeSelection() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Double>(name: "value", values: [501, nil, 500, 999, 1, 750]),
            TypedColumn<Int64>(name: "id", values: [0, 1, 2, 3, 4, 5]),
            TypedColumn<String>(name: "label", values: ["a", "b", "c", nil, "e", "f"])
        ])
        let result = try frame.filter(column: "value", where: .greaterThan(500.0))
        #expect(result.columnNames == frame.columnNames)
        #expect(result[column: "id", as: Int64.self]?.values == [0, 3, 5])
        #expect(result[column: "value", as: Double.self]?.values == [501, 999, 750])
        #expect(result[column: "value"]?.nullCount == 0)
        #expect(result[column: "label", as: String.self]?.values == ["a", nil, "f"])
        #expect(result[column: "label"]?.nullCount == 1)
        #expect(frame[column: "value"]?.nullCount == 1)
    }
}
