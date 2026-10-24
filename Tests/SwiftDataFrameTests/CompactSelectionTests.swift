import Testing
@testable import SwiftDataFrame

@Suite("Compact Double selection")
struct CompactSelectionTests {
    @Test("Gather preserves compact storage, duplicate rows and exact values across bitmap words")
    func gatherBoundaries() throws {
        let expected = (0..<130).map(fixture)
        let source = denseColumn(expected)
        for count in [0, 1, 63, 64, 65, 127, 128, 129, 261] {
            let rows = (0..<count).map { ($0 * 63 + 64) % expected.count }
            let actual = source.vGather(at: rows)
            try check(actual, expected: rows.map { expected[$0] })
            let erased = try #require(source.gathered(at: rows) as? TypedColumn<Double>)
            try check(erased, expected: rows.map { expected[$0] })
        }
        try check(source, expected: expected)
    }

    @Test("Gather retains uniform validity and drops an unnecessary bitmap")
    func uniformValidity() throws {
        for values in [[Double?](repeating: nil, count: 130), (0..<130).map { Optional(Double($0)) }] {
            let source = denseColumn(values)
            let rows = [129, 64, 63, 0, 64]
            try check(source.vGather(at: rows), expected: rows.map { values[$0] })
        }
        let present = denseColumn([nil, -0.0, .infinity]).vGather(at: [2, 1, 2])
        try check(present, expected: [.infinity, -0.0, .infinity])
        #expect(present.compactNumericStorage?.validity == nil)
    }

    @Test("Filtering, gathering and sorting keep compact columns ready for numerical consumers")
    func publicSelection() throws {
        let values = (0..<130).map(fixture)
        let identifiers = TypedColumn<Int64>(name: "id", values: (0..<130).map { Int64($0) + 9_007_199_254_740_993 })
        let compact = try DataFrame(columns: [identifiers, denseColumn(values)])
        let optional = try DataFrame(columns: [identifiers, TypedColumn(name: "x", values: values)])
        let pairs = [
            (compact.gathered(at: [129, 64, 64, 0]), optional.gathered(at: [129, 64, 64, 0])),
            (try compact.filter(column: "x", where: .greaterThan(0.0)),
             try optional.filter(column: "x", where: .greaterThan(0.0))),
            (try compact.sortBy("x"), try optional.sortBy("x")),
            (try compact.sortBy("x", ascending: false), try optional.sortBy("x", ascending: false))
        ]
        for (actual, reference) in pairs {
            let column = try #require(actual[column: "x", as: Double.self])
            let wanted = try #require(reference[column: "x", as: Double.self])
            try check(column, expected: wanted.values)
            #expect(actual[column: "id", as: Int64.self]?.values == reference[column: "id", as: Int64.self]?.values)
            let storage = try #require(column.compactNumericStorage)
            let batch = try actual.prepareNumericBatch(["x"])
            #expect(storage.values.withUnsafeBufferPointer { left in
                batch.columns[0].values.withUnsafeBufferPointer { right in left.baseAddress == right.baseAddress }
            })
        }
    }

    private func denseColumn(_ values: [Double?]) -> TypedColumn<Double> {
        var builder = CSVCompactDoubleBuilder(capacity: values.count)
        for value in values { builder.append(value) }
        return builder.column(named: "x")
    }

    private func fixture(_ row: Int) -> Double? {
        switch row % 7 {
        case 0: return nil
        case 1: return Double(bitPattern: 0x7ff8000000000042)
        case 2: return -0.0
        case 3: return .infinity
        case 4: return -.infinity
        default: return Double(row)
        }
    }

    private func check(_ actual: TypedColumn<Double>, expected: [Double?]) throws {
        let storage = try #require(actual.compactNumericStorage)
        #expect(actual.name == "x")
        #expect(actual.dtype == .float64)
        #expect(actual.count == expected.count)
        #expect(actual.nullCount == expected.filter { $0 == nil }.count)
        #expect(storage.nullCount == actual.nullCount)
        for row in expected.indices {
            #expect(actual[row]?.bitPattern == expected[row]?.bitPattern)
            #expect(storage[row]?.bitPattern == expected[row]?.bitPattern)
        }
    }
}
