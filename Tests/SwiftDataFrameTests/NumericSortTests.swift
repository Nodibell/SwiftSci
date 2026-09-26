import Foundation
import Testing
import SwiftDataFrame

@Suite("Numeric sort permutations")
struct NumericSortTests {
    private func check<T: SupportedType & Comparable>(_ values: [T?]) throws {
        let column = TypedColumn<T>(name: "key", values: values)
        for ascending in [true, false] {
            let expected = values.indices.sorted { i, j in
                switch (values[i], values[j]) {
                case let (a?, b?):
                    if a == b { return i < j }
                    return ascending ? a < b : a > b
                case (nil, nil): return i < j
                case (nil, _): return false
                case (_, nil): return true
                }
            }
            #expect(column.sortedIndices(ascending: ascending) == expected)
            let frame = try DataFrame(columns: [column,
                TypedColumn<Int64>(name: "row", values: values.indices.map { Int64($0) })])
            let sorted = try frame.sortBy("key", ascending: ascending)
            if !values.isEmpty {
                #expect(sorted[column: "row", as: Int64.self]?.values == expected.map { Int64($0) })
                #expect(sorted[column: "key"]?.nullCount == values.filter { $0 == nil }.count)
            }
        }
        #expect(column.values == values)
    }

    @Test(arguments: [0, 1, 37, 1025], [0, 1, 7])
    func stableNumericSort(count: Int, nullEvery: Int) throws {
        let values: [Int64?] = (0..<count).map {
            nullEvery > 0 && $0 % nullEvery == 0 ? nil : Int64(($0 * 37) % 19) - 9
        }
        try check(values)
        try check(values.map { $0.map { Int32($0) } })
        try check(values.map { $0.map { Double($0) } })
        try check(values.map { $0.map { Float($0) } })
    }

    @Test func exactIntegerExtremes() throws {
        try check([Int64.max, nil, Int64.min, 9_007_199_254_740_993, -9_007_199_254_740_993, Int64.max, 0])
    }

    @Test func infinitiesAndSignedZero() throws {
        try check([Double.infinity, nil, -.infinity, -0.0, 0.0, 1, -1, nil])
    }

    @Test func nanKeepsExistingComparisonBehavior() {
        for values: [Double?] in [[2, .nan, -1, .infinity, .nan], [2, nil, .nan, -1, .infinity, nil, .nan]] {
            let column = TypedColumn<Double>(name: "key", values: values)
            for ascending in [true, false] {
                let expected = values.indices.sorted { i, j in
                    switch (values[i], values[j]) {
                    case (nil, nil): return false
                    case (nil, _): return false
                    case (_, nil): return true
                    case let (a?, b?): return ascending ? a < b : a > b
                    }
                }
                #expect(column.sortedIndices(ascending: ascending) == expected)
            }
        }
    }
}
