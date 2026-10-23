import Testing
@testable import SwiftDataFrame

@Suite("Compact Double sorting")
struct CompactSortTests {
    @Test(arguments: [0, 1, 63, 64, 65, 1023, 1024, 2049, 8192])
    func finitePermutationsMatchStableOrdering(count: Int) {
        var state: UInt64 = 0x12345678
        let values = (0..<count).map { row -> Double? in
            state = state &* 6364136223846793005 &+ 1
            if row % 17 == 0 { return nil }
            if row % 19 == 0 { return row % 2 == 0 ? -0.0 : 0.0 }
            return Double(Int64(state % 1003) - 501)
        }
        checkStable(values)
        checkStable(values.sorted { ($0 ?? .infinity) < ($1 ?? .infinity) })
        checkStable(Array(values.reversed()))
    }

    @Test(arguments: [0, 1, 65, 2049])
    func sparseAndUniformColumns(count: Int) {
        checkStable([Double?](repeating: nil, count: count))
        checkStable([Double?](repeating: -0.0, count: count))
        checkStable((0..<count).map { $0 % 101 == 0 ? Double($0 % 13) : nil })
    }

    @Test(arguments: [63, 65, 2049])
    func nanFallbackKeepsExistingPermutation(count: Int) {
        var values = (0..<count).map { row -> Double? in
            row % 11 == 0 ? nil : Double((row * 73) % 997)
        }
        for row in [0, count / 2, count - 1] {
            values[row] = Double(bitPattern: 0x7ff8000000000042)
            for ascending in [false, true] {
                let optional = TypedColumn<Double>(name: "x", values: values)
                #expect(compact(values).sortedIndices(ascending: ascending) == optional.sortedIndices(ascending: ascending))
            }
        }
    }

    @Test(arguments: [false, true])
    func publicSortPreservesValueBitsAndNumericHandoff(ascending: Bool) throws {
        let values: [Double?] = [nil, -0.0, .infinity, 0.0, -.infinity, 4, -2, 4, nil]
        let identifiers = TypedColumn<Int64>(name: "id", values: values.indices.map { Int64($0) + 9_007_199_254_740_993 })
        let frame = try DataFrame(columns: [identifiers, compact(values)])
        let sorted = try frame.sortBy("x", ascending: ascending)
        let expected = stableRows(values, ascending: ascending)
        let column = try #require(sorted[column: "x", as: Double.self])
        let storage = try #require(column.compactNumericStorage)
        #expect(column.values.map { $0?.bitPattern } == expected.map { values[$0]?.bitPattern })
        #expect(sorted[column: "id", as: Int64.self]?.values == expected.map { identifiers[$0] })
        #expect(column.nullCount == 2)
        let batch = try sorted.prepareNumericBatch(["x"])
        #expect(storage.values.withUnsafeBufferPointer { left in
            batch.columns[0].values.withUnsafeBufferPointer { right in left.baseAddress == right.baseAddress }
        })
        #expect(frame[column: "x", as: Double.self]?.values.map { $0?.bitPattern } == values.map { $0?.bitPattern })
    }

    private func checkStable(_ values: [Double?]) {
        let column = compact(values)
        for ascending in [false, true] {
            #expect(column.sortedIndices(ascending: ascending) == stableRows(values, ascending: ascending))
        }
    }

    private func stableRows(_ values: [Double?], ascending: Bool) -> [Int] {
        values.indices.sorted { left, right in
            switch (values[left], values[right]) {
            case (nil, nil): return left < right
            case (nil, _): return false
            case (_, nil): return true
            case let (a?, b?):
                if a == b { return left < right }
                return ascending ? a < b : a > b
            }
        }
    }

    private func compact(_ values: [Double?]) -> TypedColumn<Double> {
        var builder = CSVCompactDoubleBuilder(capacity: values.count)
        for value in values { builder.append(value) }
        return builder.column(named: "x")
    }
}
