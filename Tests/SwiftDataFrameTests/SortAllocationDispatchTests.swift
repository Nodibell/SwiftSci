import Testing
@testable import SwiftDataFrame

@Suite("Allocation-free Double sort dispatch")
struct SortAllocationDispatchTests {
    @Test(arguments: [16, 64, 256], [false, true])
    func orderedRunsKeepStablePermutations(runLength: Int, compact: Bool) {
        let values: [Double?] = (0..<8193).map { row in
            if row % 31 == 0 { return nil }
            let block = row / runLength
            return Double(block * runLength + (block % 2 == 0 ? row % runLength : runLength - row % runLength))
        }
        let column = makeColumn(values, compact: compact)
        for ascending in [false, true] {
            let expected = values.indices.sorted { left, right in
                switch (values[left], values[right]) {
                case (nil, nil): return left < right
                case (nil, _): return false
                case (_, nil): return true
                case let (a?, b?): return a == b ? left < right : ascending ? a < b : a > b
                }
            }
            #expect(column.sortedIndices(ascending: ascending) == expected)
        }
        #expect(column.values.map { $0?.bitPattern } == values.map { $0?.bitPattern })
    }

    @Test(arguments: [0, 129, 8192], [false, true])
    func nanBeforeOrAfterDispatchKeepsComparisonFallback(position: Int, compact: Bool) {
        var values: [Double?] = (0..<8193).map { (row: Int) -> Double? in
            if row % 31 == 0 { return nil }
            return Double((row * 7919) % 4093)
        }
        values[position] = Double(bitPattern: 0x7ff8000000000042)
        let column = makeColumn(values, compact: compact)
        for ascending in [false, true] {
            let expected = values.indices.sorted { left, right in
                switch (values[left], values[right]) {
                case (nil, nil): return false
                case (nil, _): return false
                case (_, nil): return true
                case let (a?, b?): return ascending ? a < b : a > b
                }
            }
            #expect(column.sortedIndices(ascending: ascending) == expected)
        }
        #expect(column.values.map { $0?.bitPattern } == values.map { $0?.bitPattern })
    }

    private func makeColumn(_ values: [Double?], compact: Bool) -> TypedColumn<Double> {
        guard compact else { return TypedColumn(name: "x", values: values) }
        var builder = CSVCompactDoubleBuilder(capacity: values.count)
        for value in values { builder.append(value) }
        return builder.column(named: "x")
    }
}
