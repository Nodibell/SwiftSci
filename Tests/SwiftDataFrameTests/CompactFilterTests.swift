import Testing
@testable import SwiftDataFrame

@Suite("Compact Double predicate filtering")
struct CompactFilterTests {
    @Test(arguments: [0, 1, 63, 64, 65, 127, 128, 129, 2049])
    func comparisonsMatchIEEEValues(count: Int) {
        var state: UInt64 = 0x12345678
        var values = (0..<count).map { row -> Double? in
            state = state &* 6364136223846793005 &+ 1
            return row % 17 == 0 ? nil : Double(bitPattern: state)
        }
        let special: [Double?] = [nil, .nan, -.infinity, .infinity, -0.0, 0.0,
                                  .leastNonzeroMagnitude, .greatestFiniteMagnitude]
        for row in 0..<min(values.count, special.count) { values[row] = special[row] }
        let column = compact(values)
        let thresholds: [Double] = [.nan, -.infinity, .infinity, -0.0, 0.1,
                                    1.0.nextDown, 1.0, 1.0.nextUp]
        for rhs in thresholds {
            let predicates: [(Double) -> Bool] = [
                { $0 == rhs }, { $0 != rhs }, { $0 < rhs },
                { $0 <= rhs }, { $0 > rhs }, { $0 >= rhs }
            ]
            for (condition, predicate) in zip(conditions(rhs), predicates) {
                let expected = values.indices.filter { values[$0].map(predicate) ?? false }
                #expect(column.filteredIndices(matching: condition) == expected)
            }
        }
        #expect(column.filteredIndices(matching: .isNull) == values.indices.filter { values[$0] == nil })
        #expect(column.filteredIndices(matching: .isNotNull) == values.indices.filter { values[$0] != nil })
    }

    @Test(arguments: [false, true])
    func exactIntegerThresholds(includeNulls: Bool) {
        let lower = -9_223_372_036_854_775_808.0
        let upper = -lower
        let exact = 9_007_199_254_740_992.0
        let cases: [(Int64, [Double], [[Int]])] = [
            (9_007_199_254_740_993, [exact, exact.nextUp, -exact, -exact.nextUp],
             [[], [0, 1, 2, 3], [0, 2, 3], [0, 2, 3], [1], [1]]),
            (Int64.max, [upper, upper.nextDown, lower],
             [[], [0, 1, 2], [1, 2], [1, 2], [0], [0]]),
            (Int64.min + 1, [lower, lower.nextUp],
             [[], [0, 1], [0], [0], [1], [1]])
        ]
        for (threshold, present, expected) in cases {
            var values = present.map(Optional.some)
            if includeNulls { values.append(nil) }
            let column = compact(values)
            for (condition, rows) in zip(conditions(threshold), expected) {
                #expect(column.filteredIndices(matching: condition) == rows)
            }
        }
    }

    @Test(arguments: [false, true])
    func publicFiltersPreserveSelectedBitsAndNumericHandoff(includeNulls: Bool) throws {
        var values: [Double?] = [-0.0, 0.0, .nan, -.infinity, .infinity, 1, -1]
        if includeNulls { values.insert(nil, at: 0) }
        let frame = try DataFrame(columns: [compact(values)])
        let conditions: [FilterCondition] = [.notEquals(0.0), .greaterThan(0.0), .isNull, .isNotNull]
        let optional = TypedColumn<Double>(name: "x", values: values)
        for condition in conditions {
            let rows = try #require(optional.filteredIndices(matching: condition))
            for filtered in [try frame.filter(column: "x", where: condition),
                             try frame.filterFast(column: "x", where: condition)] {
                let column = try #require(filtered[column: "x", as: Double.self])
                let storage = try #require(column.compactNumericStorage)
                #expect(column.values.map { $0?.bitPattern } == rows.map { values[$0]?.bitPattern })
                #expect(column.nullCount == rows.filter { values[$0] == nil }.count)
                let batch = try filtered.prepareNumericBatch(["x"])
                #expect(storage.values.withUnsafeBufferPointer { left in
                    batch.columns[0].values.withUnsafeBufferPointer { right in left.baseAddress == right.baseAddress }
                })
            }
        }
    }

    private func compact(_ values: [Double?]) -> TypedColumn<Double> {
        var builder = CSVCompactDoubleBuilder(capacity: values.count)
        for value in values { builder.append(value) }
        return builder.column(named: "x")
    }

    private func conditions(_ value: any Sendable) -> [FilterCondition] {
        [.equals(value), .notEquals(value), .lessThan(value), .lessThanOrEqual(value),
         .greaterThan(value), .greaterThanOrEqual(value)]
    }
}
