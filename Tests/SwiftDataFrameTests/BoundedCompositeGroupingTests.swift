import Foundation
import Testing
import SwiftDataFrame

@Suite("Bounded composite integer grouping")
struct BoundedCompositeGroupingTests {
    private func checkPairs<T: FixedWidthInteger & SupportedType>(
        _ first: [T?], _ second: [T?]
    ) throws {
        var representatives: [Int] = []
        var sums: [Double] = []
        var counts: [Int64] = []
        var expectedGroups: [Int] = []
        for row in first.indices {
            let found = representatives.firstIndex {
                first[$0] == first[row] && second[$0] == second[row]
            }
            let group = found ?? representatives.count
            if found == nil {
                representatives.append(row)
                sums.append(0)
                counts.append(0)
            }
            sums[group] += Double(row + 1)
            counts[group] += 1
            expectedGroups.append(group)
        }
        let frame = try DataFrame(columns: [
            TypedColumn<T>(name: "a", values: first),
            TypedColumn<T>(name: "b", values: second),
            TypedColumn<Double>(name: "value", values: first.indices.map { Double($0 + 1) })])
        let grouped = frame.groupBy("a", "b")
        let result = grouped.sum()
        #expect(result[column: "a", as: String.self]?.values == representatives.map { first[$0].map { String($0) } })
        #expect(result[column: "b", as: String.self]?.values == representatives.map { second[$0].map { String($0) } })
        #expect(result[column: "value", as: Double.self]?.values == sums.map(Optional.init))
        #expect(grouped.count()[column: "value", as: Int64.self]?.values == counts.map(Optional.init))
        #expect(grouped.transform(["value": .sum])[column: "value_group_sum", as: Double.self]?.values == expectedGroups.map { Optional(sums[$0]) })
    }

    @Test func denseDomainsPreserveOrderAndNullIdentityAcrossIntegerWidths() throws {
        let first: [Int?] = (0..<192).map { [2, nil, 0, 1][$0 % 4] }
        let second: [Int?] = (0..<192).map { [nil, -1, 1, 0, -1][($0 / 3) % 5] }
        try checkPairs(first, second)
        try checkPairs(first.map { $0.map(Int32.init) }, second.map { $0.map(Int32.init) })
        try checkPairs(first.map { $0.map(Int64.init) }, second.map { $0.map(Int64.init) })
        try checkPairs(second, first)
    }

    @Test func compactRangesAtIntegerLimitsStayExact() throws {
        let first: [Int64?] = (0..<64).map { $0 % 11 == 0 ? nil : Int64.max - Int64($0 % 3) }
        let second: [Int64?] = (0..<64).map { $0 % 7 == 0 ? nil : Int64.min + Int64(($0 / 3) % 2) }
        try checkPairs(first, second)
        try checkPairs(first.map { $0.map(Int.init) }, second.map { $0.map(Int.init) })
        let narrowFirst: [Int32?] = (0..<64).map { Int32.max - Int32($0 % 3) }
        let narrowSecond: [Int32?] = (0..<64).map { Int32.min + Int32(($0 / 3) % 2) }
        try checkPairs(narrowFirst, narrowSecond)
    }

    @Test func sparseAndOverflowingSpansUseExactFallback() throws {
        let first: [Int64?] = [0, 1, 0, 1, 0, 1, nil, nil]
        let second: [Int64?] = [Int64.min, Int64.max, Int64.max, Int64.min, Int64.min, Int64.max, 0, nil]
        try checkPairs(first, second)
        try checkPairs(second, first)
        let high = Int64(9_007_199_254_740_992)
        try checkPairs(first, [high, high + 1, high + 1, high, high, high + 1, nil, nil])
        try checkPairs(first, [0, 1_000_000, 0, 1_000_000, nil, nil, 0, 1_000_000])
    }

    @Test func allNullComponentPreservesPriorGroups() throws {
        let keys: [Int64?] = [2, 1, nil, 2, 0, 1, nil, 0]
        let missing = [Int64?](repeating: nil, count: keys.count)
        try checkPairs(keys, missing)
        try checkPairs(missing, keys)
        try checkPairs(missing, missing)
    }

    @Test func domainProductLargerThanRowBudgetFallsBack() throws {
        let first: [Int?] = (0..<128).map { $0 % 16 }
        let second: [Int?] = (0..<128).map { ($0 * 7) % 31 }
        try checkPairs(first, second)
    }

    @Test func nullSlotCanPushDomainProductPastBudget() throws {
        let first: [Int?] = (0..<16).map { $0 % 4 }
        let second: [Int?] = (0..<16).map { $0 == 0 ? nil : $0 / 4 }
        try checkPairs(first, second)
    }

    @Test func domainProductLargerThanByteBudgetFallsBack() throws {
        let distinct = 257 * 257
        let rows = distinct + 1
        let frame = try DataFrame(columns: [
            TypedColumn<Int>(name: "a", values: (0..<rows).map { $0 % 257 }),
            TypedColumn<Int>(name: "b", values: (0..<rows).map { ($0 / 257) % 257 }),
            TypedColumn<Int64>(name: "value", values: Array(repeating: 1, count: rows))])
        let result = try frame.groupBy("a", "b").sumChecked()
        #expect(result.shape.rows == distinct)
        #expect(result[column: "a", as: String.self]?.values == (0..<distinct).map { Optional(String($0 % 257)) })
        #expect(result[column: "b", as: String.self]?.values == (0..<distinct).map { Optional(String($0 / 257)) })
        #expect(result[column: "value", as: Int64.self]?.values == [Optional(Int64(2))] + Array(repeating: Optional(Int64(1)), count: distinct - 1))
    }

    @Test func hypotheticalCartesianProductCannotOverflowGroupIdentity() throws {
        let values: [Int64] = (0..<1_000).map(Int64.init)
        var columns: [any AnyColumn] = (0..<7).map { key in
            TypedColumn<Int64>(name: "key\(key)", values: values)
        }
        columns.append(TypedColumn<Int64>(name: "value", values: Array(repeating: 1, count: values.count)))
        let frame = try DataFrame(columns: columns)
        let result = try frame.groupBy("key0", "key1", "key2", "key3", "key4", "key5", "key6").sumChecked()
        #expect(result.shape.rows == values.count)
        #expect(result[column: "value", as: Int64.self]?.values == Array(repeating: Optional(Int64(1)), count: values.count))
        for key in 0..<7 {
            #expect(result[column: "key\(key)", as: String.self]?.values == values.map { Optional(String($0)) })
        }
    }

    @Test func unicodeEqualitySurvivesIntegerRefinement() throws {
        let composed = "\u{e9}"
        let decomposed = "e\u{301}"
        let frame = try DataFrame(columns: [
            TypedColumn<String>(name: "text", values: [composed, decomposed, "other", composed, decomposed, nil]),
            TypedColumn<Int>(name: "number", values: [1, 1, 1, 2, 2, 1]),
            TypedColumn<Double>(name: "value", values: [1, 2, 3, 4, 5, 6])])
        for result in [frame.groupBy("text", "number").sum(), frame.groupBy("number", "text").sum()] {
            #expect(result[column: "text", as: String.self]?.values == [composed, "other", composed, nil])
            #expect(result[column: "number", as: String.self]?.values == ["1", "1", "2", "1"])
            #expect(result[column: "value", as: Double.self]?.values == [3, 3, 9, 6])
        }
    }
}
