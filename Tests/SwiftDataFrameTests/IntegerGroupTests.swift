import Testing
import SwiftDataFrame

@Suite("Integer grouping semantics")
struct IntegerGroupTests {
    @Test func extremaNullsAndAggregations() throws {
        let large: Int64 = 9_007_199_254_740_993
        let keys: [Int64?] = [.max, nil, .min, large, .max, large, nil, .min]
        let frame = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: keys),
            TypedColumn<Double>(name: "value", values: [1, nil, 3, nil, 5, 6, nil, 8])])
        let grouped = frame.groupBy("key")
        let expectedKeys: [String?] = [String(Int64.max), nil, String(Int64.min), String(large)]
        for result in [grouped.sum(), grouped.mean(), grouped.min(), grouped.max(), grouped.count()] {
            #expect(result[column: "key", as: String.self]?.values == expectedKeys)
        }
        #expect(grouped.sum()[column: "value", as: Double.self]?.values == [6, nil, 11, 6])
        #expect(grouped.mean()[column: "value", as: Double.self]?.values == [3, nil, 5.5, 6])
        #expect(grouped.min()[column: "value", as: Double.self]?.values == [1, nil, 3, 6])
        #expect(grouped.max()[column: "value", as: Double.self]?.values == [5, nil, 8, 6])
        #expect(grouped.count()[column: "value", as: Int64.self]?.values == [2, 2, 2, 2])
        #expect(grouped.agg(["value": .count])[column: "value_count", as: Double.self]?.values == [2, 0, 2, 1])
        #expect(grouped.agg(["value": .first])[column: "value_first", as: Double.self]?.values == [1, nil, 3, 6])
        #expect(grouped.agg(["value": .last])[column: "value_last", as: Double.self]?.values == [5, nil, 8, 6])
        #expect(grouped.transform(["value": .sum])[column: "value_group_sum", as: Double.self]?.values == [6, nil, 11, 6, 6, 6, nil, 11])
        #expect(frame[column: "key", as: Int64.self]?.values == keys)
    }

    private func check<T: SupportedType>(_ keys: [T?], expected: [String?], counts: [Int64?]) throws {
        let frame = try DataFrame(columns: [TypedColumn<T>(name: "key", values: keys),
            TypedColumn<Double>(name: "value", values: Array(repeating: 1, count: keys.count))])
        let result = frame.groupBy("key").count()
        #expect(result[column: "key", as: String.self]?.values == expected)
        #expect(result[column: "value", as: Int64.self]?.values == counts)
    }

    @Test func integerWidthsAndEmptyInputs() throws {
        try check([Int32(7), nil, -3, 7, nil], expected: ["7", nil, "-3"], counts: [2, 2, 1])
        try check([Int(7), nil, -3, 7, nil], expected: ["7", nil, "-3"], counts: [2, 2, 1])
        try check([Int64?](repeating: nil, count: 4), expected: [nil], counts: [4])
        try check([Int64?](), expected: [], counts: [])
        try check((0..<257).map { Int64($0) }, expected: (0..<257).map { String($0) }, counts: Array(repeating: 1, count: 257))
    }

    @Test func multipleKeysKeepTheirExistingGrouping() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Int64>(name: "key", values: [1, 1, 2, 1]),
            TypedColumn<String>(name: "label", values: ["a", "b", "a", "a"]),
            TypedColumn<Double>(name: "value", values: [2, 3, 4, 5])])
        let result = frame.groupBy("key", "label").sum()
        #expect(result[column: "key", as: String.self]?.values == ["1", "1", "2"])
        #expect(result[column: "label", as: String.self]?.values == ["a", "b", "a"])
        #expect(result[column: "value", as: Double.self]?.values == [7, 3, 4])
    }
}
