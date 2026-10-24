import Testing
@testable import SwiftDataFrame

@Suite("Stable monotonic Double permutations")
struct MonotonicDoubleSortingTests {
    @Test(arguments: [-1, 0, 1], [false, true])
    func monotonicTiesKeepInputOrder(direction: Int, compact: Bool) {
        for missing in [false, true] {
            let values: [Double?] = (0..<4097).map { row in
                if missing && row % 31 == 0 { return nil }
                return Double(direction * (row / 7 - 256))
            }
            check(values, compact: compact)
        }
    }

    @Test(arguments: [false, true], [false, true])
    func zerosInfinitiesAndMissingKeepTheirBits(compact: Bool, reversed: Bool) {
        var values: [Double?] = (0..<4097).map { row in
            if row % 31 == 0 { return nil }
            if row < 1024 { return -.infinity }
            if row < 3072 { return row % 2 == 0 ? -0.0 : 0.0 }
            return .infinity
        }
        if reversed { values.reverse() }
        check(values, compact: compact)
    }

    private func check(_ values: [Double?], compact: Bool) {
        let column: TypedColumn<Double>
        if compact {
            var builder = CSVCompactDoubleBuilder(capacity: values.count)
            for value in values { builder.append(value) }
            column = builder.column(named: "x")
        } else {
            column = TypedColumn(name: "x", values: values)
        }
        for ascending in [false, true] {
            let expected = values.indices.sorted { left, right in
                switch (values[left], values[right]) {
                case (nil, nil): return left < right
                case (nil, _): return false
                case (_, nil): return true
                case let (a?, b?): return a == b ? left < right : ascending ? a < b : a > b
                }
            }
            let actual = column.sortedIndices(ascending: ascending)
            #expect(actual == expected)
            #expect(actual.map { values[$0]?.bitPattern } == expected.map { values[$0]?.bitPattern })
        }
        #expect(column.values.map { $0?.bitPattern } == values.map { $0?.bitPattern })
    }
}
