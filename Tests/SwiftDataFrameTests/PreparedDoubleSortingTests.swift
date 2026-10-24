import Testing
@testable import SwiftDataFrame

@Suite("Prepared Double sorting contracts")
struct PreparedDoubleSortingTests {
    @Test(arguments: [3, 4095, 8187], [false, true])
    func firstTurnPreservesStableRows(position: Int, compact: Bool) throws {
        for missing in [false, true] {
            for reversed in [false, true] {
                var values: [Double?] = (0..<8193).map { (row: Int) -> Double? in
                    if missing && row % 31 == 0 { return nil }
                    return Double(row / 4 - 1024)
                }
                if reversed { values.reverse() }
                let turn = reversed ? position + 1 : position
                values.swapAt(turn, turn + 1)
                try check(values, compact: compact)
            }
        }
    }

    @Test(arguments: [2047, 2048, 2049], [false, true])
    func missingDensityPreservesStableRows(validCount: Int, compact: Bool) throws {
        let rows = (0..<validCount).map { $0 * 4096 / validCount }
        var values = [Double?](repeating: nil, count: 4096)
        for (rank, row) in rows.enumerated() { values[row] = Double(rank / 4 - 256) }
        values.swapAt(rows[3], rows[4])
        try check(values, compact: compact)
    }

    @Test(arguments: [false, true])
    func nearlyOrderedFloatingEdgesRetainBits(compact: Bool) throws {
        let edges: [Double?] = [-.infinity, -.greatestFiniteMagnitude, -1,
            -.leastNormalMagnitude, -.leastNonzeroMagnitude, -0.0, 0.0,
            .leastNonzeroMagnitude, .leastNormalMagnitude, 1, .greatestFiniteMagnitude, .infinity, nil]
        var values = (0..<4096).map { edges[$0 * edges.count / 4096] }
        values.swapAt(1023, 3071)
        try check(values, compact: compact)
    }

    private func check(_ values: [Double?], compact: Bool) throws {
        let column: TypedColumn<Double>
        if compact {
            var builder = CSVCompactDoubleBuilder(capacity: values.count)
            for value in values { builder.append(value) }
            column = builder.column(named: "x")
        } else {
            column = TypedColumn(name: "x", values: values)
        }
        let identifiers = TypedColumn<Int64>(name: "id",
            values: values.indices.map { Int64($0) + 9_007_199_254_740_993 })
        let frame = try DataFrame(columns: [column, identifiers])
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
            let sorted = try frame.sortBy("x", ascending: ascending)
            #expect(sorted[column: "id", as: Int64.self]?.values == expected.map { identifiers[$0] })
            #expect(sorted[column: "x", as: Double.self]?.values.map { $0?.bitPattern }
                    == expected.map { values[$0]?.bitPattern })
        }
        #expect(column.values.map { $0?.bitPattern } == values.map { $0?.bitPattern })
    }
}
