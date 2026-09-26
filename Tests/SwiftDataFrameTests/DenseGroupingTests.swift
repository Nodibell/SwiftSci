import Testing
import SwiftDataFrame

@Suite("Grouped reduction order and missing values")
struct DenseGroupingTests {
    private func check(_ key: any AnyColumn) throws {
        let values: [Double?] = [1e16, 5, 1, nil, -1e16, nil, nil, -2, 3, 4]
        let frame = try DataFrame(columns: [key, TypedColumn<Double>(name: "value", values: values)])
        let group = frame.groupBy("key")
        #expect(group.sum()[column: "value", as: Double.self]?.values == [1, 3, nil, 7])
        #expect(group.mean()[column: "value", as: Double.self]?.values == [1.0 / 3, 1.5, nil, 3.5])
        #expect(group.min()[column: "value", as: Double.self]?.values == [-1e16, -2, nil, 3])
        #expect(group.max()[column: "value", as: Double.self]?.values == [1e16, 5, nil, 4])
        #expect(group.count()[column: "value", as: Int64.self]?.values == [3, 3, 2, 2])
        #expect(group.agg(["value": .count])[column: "value_count", as: Double.self]?.values == [3, 2, 0, 2])
        #expect(group.agg(["value": .first])[column: "value_first", as: Double.self]?.values == [1e16, 5, nil, 3])
        #expect(group.agg(["value": .last])[column: "value_last", as: Double.self]?.values == [-1e16, -2, nil, 4])
        #expect(group.transform(["value": .sum])[column: "value_group_sum", as: Double.self]?.values == [1, 3, 1, 3, 1, nil, nil, 3, 7, 7])
    }

    @Test func cancellationWithInterleavedGroups() throws {
        try check(TypedColumn<Int64>(name: "key", values: [0, 1, 0, 1, 0, 2, 2, 1, nil, nil]))
        try check(TypedColumn<String>(name: "key", values: ["a", "b", "a", "b", "a", "c", "c", "b", nil, nil]))
    }

    @Test func numericValueWidths() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: [2, 1, 2, 1]),
            TypedColumn<Float>(name: "float", values: [1, nil, 3, nil]),
            TypedColumn<Int32>(name: "int32", values: [1, nil, 3, nil]),
            TypedColumn<Int64>(name: "int64", values: [1, nil, 3, nil])])
        for name in ["float", "int32", "int64"] {
            #expect(frame.groupBy("key").sum()[column: name, as: Double.self]?.values == [4, nil])
            #expect(frame.groupBy("key").mean()[column: name, as: Double.self]?.values == [2, nil])
            #expect(frame.groupBy("key").agg([name: .count])[column: name + "_count", as: Double.self]?.values == [2, 0])
        }
    }

    @Test func nanAndInfinityOrder() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: [0, 1, 0, 1, 1, 2, 2]),
            TypedColumn<Double>(name: "value", values: [.nan, 2, 5, .nan, -3, .infinity, -.infinity])])
        let group = frame.groupBy("key")
        let sums = try #require(group.sum()[column: "value", as: Double.self])
        #expect(sums.values.allSatisfy { $0?.isNaN == true })
        let minima = try #require(group.min()[column: "value", as: Double.self])
        #expect(minima.values[0]?.isNaN == true)
        #expect(minima.values[1] == -3)
        #expect(minima.values[2] == -.infinity)
        let first = try #require(group.agg(["value": .first])[column: "value_first", as: Double.self])
        #expect(first.values[0]?.isNaN == true)
        #expect(first.values[1] == 2)
        #expect(group.agg(["value": .last])[column: "value_last", as: Double.self]?.values == [5, -3, -.infinity])
    }

    @Test func emptyKeySelectionAndLegacyStringNullKey() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<String>(name: "key", values: ["__null__", nil, "b"]),
            TypedColumn<Double>(name: "value", values: [1, 2, 3])])
        let grouped = frame.groupBy("key").sum()
        #expect(grouped[column: "key", as: String.self]?.values == ["__null__", "b"])
        #expect(grouped[column: "value", as: Double.self]?.values == [3, 3])
        #expect(frame.groupBy().sum().rowCount == 0)
        #expect(frame.groupBy().transform(["value": .sum])[column: "value_group_sum", as: Double.self]?.values == [nil, nil, nil])
        #expect(frame.groupBy("absent").sum()[column: "value", as: Double.self]?.values == [6])
    }
}
