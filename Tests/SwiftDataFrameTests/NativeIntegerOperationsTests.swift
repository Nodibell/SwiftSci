import Testing
import SwiftDataFrame

@Suite("Native integer operations")
struct NativeIntegerOperationsTests {
    @Test func sortingPreservesIntegerPrecision() {
        let values: [Int?] = [9_007_199_254_740_993, 9_007_199_254_740_992, nil, .max, .min, 9_007_199_254_740_993]
        let column = TypedColumn<Int>(name: "value", values: values)
        #expect(column.sortedIndices(ascending: true) == [4, 1, 0, 5, 3, 2])
        #expect(column.sortedIndices(ascending: false) == [3, 0, 5, 1, 4, 2])
    }

    @Test func reductionsMatchOtherIntegerWidths() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Int>(name: "key", values: [0, 1, 0, 1, 2]),
            TypedColumn<Int>(name: "value", values: [4, nil, -2, nil, 7])])
        let grouped = frame.groupBy("key")
        #expect(grouped.sum()[column: "value", as: Double.self]?.values == [2, nil, 7])
        #expect(grouped.mean()[column: "value", as: Double.self]?.values == [1, nil, 7])
        #expect(grouped.min()[column: "value", as: Double.self]?.values == [-2, nil, 7])
        #expect(grouped.max()[column: "value", as: Double.self]?.values == [4, nil, 7])
        #expect(grouped.agg(["value": .count])[column: "value_count", as: Double.self]?.values == [2, 0, 1])
        #expect(grouped.agg(["value": .first])[column: "value_first", as: Double.self]?.values == [4, nil, 7])
        #expect(grouped.agg(["value": .last])[column: "value_last", as: Double.self]?.values == [-2, nil, 7])
        #expect(grouped.transform(["value": .sum])[column: "value_group_sum", as: Double.self]?.values == [2, nil, 2, nil, 7])
    }

    @Test func customNativeIntegerValuesAreNumeric() throws {
        let value = DataFrameRelease35CoverageTests.CustomGenericColumn(
            name: "value", dtype: .int64, rawValues: [Int(4), Int(-2), nil])
        let frame = try DataFrame(columns: [
            TypedColumn<Int>(name: "key", values: [0, 0, 1]), value])
        #expect(frame.groupBy("key").sum()[column: "value", as: Double.self]?.values == [2, nil])
    }
}
