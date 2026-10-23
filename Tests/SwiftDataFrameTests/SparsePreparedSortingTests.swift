import Testing
@testable import SwiftDataFrame

@Suite("Sparse prepared sorting contracts")
struct SparsePreparedSortingTests {
    @Test(arguments: [1638, 1724, 3276, 8192, 16056], [false, true])
    func firstTurnPreservesSparseRows(validCount: Int, compact: Bool) throws {
        for clustered in [false, true] {
            let rows = presentRows(validCount, clustered: clustered)
            for turn in [3, (validCount / 2 / 4) * 4 - 1, (validCount / 4 - 1) * 4 - 1] {
                var values = [Double?](repeating: nil, count: 32768)
                for (rank, row) in rows.enumerated() { values[row] = Double(rank / 4 - 2048) }
                values.swapAt(rows[turn], rows[turn + 1])
                try check(values, compact: compact, legacyNaN: false)
            }
        }
    }

    @Test(arguments: [1724, 8192], [false, true])
    func sparseNaNRetainsComparisonFallback(validCount: Int, compact: Bool) throws {
        for clustered in [false, true] {
            let rows = presentRows(validCount, clustered: clustered)
            for rank in [0, validCount / 2, validCount - 1] {
                var values = [Double?](repeating: nil, count: 32768)
                for (rank, row) in rows.enumerated() { values[row] = Double(rank / 4) }
                values.swapAt(rows[3], rows[4])
                values[rows[rank]] = Double(bitPattern: 0x7ff8000000000042)
                try check(values, compact: compact, legacyNaN: true)
            }
        }
    }

    private func presentRows(_ count: Int, clustered: Bool) -> [Int] {
        (0..<count).map { clustered ? 8192 + $0 : $0 * 32768 / count }
    }

    private func check(_ values: [Double?], compact: Bool, legacyNaN: Bool) throws {
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
                case (nil, nil): return legacyNaN ? false : left < right
                case (nil, _): return false
                case (_, nil): return true
                case let (a?, b?):
                    if !legacyNaN && a == b { return left < right }
                    return ascending ? a < b : a > b
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
