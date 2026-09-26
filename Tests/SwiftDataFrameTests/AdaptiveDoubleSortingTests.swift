import Foundation
import Testing
import SwiftDataFrame

@Suite("Adaptive Double sorting")
struct AdaptiveDoubleSortingTests {
    private func check(_ values: [Double?]) throws {
        let column = TypedColumn<Double>(name: "key", values: values)
        let frame = try DataFrame(columns: [column,
            TypedColumn<Int64>(name: "row", values: values.indices.map { Int64($0) })])
        for ascending in [true, false] {
            let expected = values.indices.sorted { a, b in
                switch (values[a], values[b]) {
                case (nil, nil): return a < b
                case (nil, _): return false
                case (_, nil): return true
                case let (x?, y?):
                    if x == y { return a < b }
                    return ascending ? x < y : x > y
                }
            }
            #expect(column.sortedIndices(ascending: ascending) == expected)
            let result = try frame.sortBy("key", ascending: ascending)
            if values.isEmpty { continue }
            #expect(result[column: "row", as: Int64.self]?.values == expected.map { Int64($0) })
            let sorted = try #require(result[column: "key", as: Double.self])
            #expect(sorted.nullCount == column.nullCount)
            #expect(sorted.values.map { $0?.bitPattern } == expected.map { values[$0]?.bitPattern })
        }
        #expect(column.values.map { $0?.bitPattern } == values.map { $0?.bitPattern })
    }

    @Test(arguments: [0, 1, 1_023, 1_024, 1_025, 4_097], [0, 3, 17])
    func stableDisorderedTies(count: Int, nullEvery: Int) throws {
        let values: [Double?] = (0..<count).map { index in
            nullEvery > 0 && index % nullEvery == 0 ? nil : Double((index * 7919) % 101) - 50
        }
        try check(values)
    }

    @Test func floatingBoundariesPreservePayloads() throws {
        let special: [Double?] = [0, -0.0, .infinity, -.infinity,
            .leastNonzeroMagnitude, -.leastNonzeroMagnitude,
            .leastNormalMagnitude, -.leastNormalMagnitude,
            .greatestFiniteMagnitude, -.greatestFiniteMagnitude, nil, 1, -1]
        try check((0..<4_097).map { special[($0 * 7) % special.count] })
    }

    @Test func orderedAndNearlyOrderedRuns() throws {
        let ordered: [Double?] = (0..<4_096).map { Double($0 / 4) }
        try check(ordered)
        try check(Array(ordered.reversed()))
        var nearlyOrdered = ordered
        for index in stride(from: 511, to: nearlyOrdered.count - 1, by: 512) {
            nearlyOrdered.swapAt(index, index + 1)
        }
        try check(nearlyOrdered)
        try check(Array(nearlyOrdered.reversed()))
        var displaced = ordered
        displaced.insert(displaced.removeLast(), at: 17)
        try check(displaced)
        let runs: [Double?] = (0..<4_096).map { index in
            let block = index / 256
            return Double(block * 256 + (block % 2 == 0 ? index % 256 : 255 - index % 256))
        }
        try check(runs)
    }

    @Test func sparseColumnsPreserveStableRows() throws {
        for stride in [10, 50] {
            let values: [Double?] = (0..<60_000).map { index in
                index % stride == 0 ? Double((index * 37) % 101) : nil
            }
            try check(values)
        }
    }

    @Test func missingAndEqualValues() throws {
        try check([Double?](repeating: nil, count: 2_048))
        try check((0..<2_048).map { $0 % 2 == 0 ? 0 : -0.0 })
    }

    @Test(arguments: [1, 2_048])
    func nanRetainsComparisonFallback(nanIndex: Int) {
        var values: [Double?] = (0..<2_049).map { $0 % 19 == 0 ? nil : Double(($0 * 37) % 101) }
        values[nanIndex] = Double(bitPattern: 0xfff8_0000_0000_0001)
        let column = TypedColumn<Double>(name: "key", values: values)
        for ascending in [true, false] {
            let expected = values.indices.sorted { a, b in
                switch (values[a], values[b]) {
                case (nil, nil): return false
                case (nil, _): return false
                case (_, nil): return true
                case let (x?, y?): return ascending ? x < y : x > y
                }
            }
            #expect(column.sortedIndices(ascending: ascending) == expected)
        }
    }
}
