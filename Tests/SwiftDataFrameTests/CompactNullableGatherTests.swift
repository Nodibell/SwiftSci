import Testing
@testable import SwiftDataFrame

@Suite("Compact nullable gathering")
struct CompactNullableGatherTests {
    @Test("Missing row indices and missing values retain separate validity across bitmap words")
    func nullableRows() throws {
        let values = (0..<130).map(fixture)
        let source = dense(values, name: "x")
        for count in [0, 1, 63, 64, 65, 129, 261] {
            let rows: [Int?] = (0..<count).map { $0 % 5 == 0 ? nil : ($0 * 63 + 64) % values.count }
            let actual = try #require(source.gatheredWithNulls(at: rows, newName: "selected") as? TypedColumn<Double>)
            try check(actual, rows.map { $0.flatMap { values[$0] } }, name: "selected")
        }
        let empty = dense([], name: "x")
        let missing = try #require(empty.gatheredWithNulls(at: [nil, nil]) as? TypedColumn<Double>)
        try check(missing, [nil, nil], name: "x")
        let present = try #require(source.gatheredWithNulls(at: [2, 3, 2]) as? TypedColumn<Double>)
        try check(present, [values[2], values[3], values[2]], name: "x")
        #expect(present.compactNumericStorage?.validity == nil)
        try check(source, values, name: "x")
    }

    @Test("Joined keys prefer a mapped left row even when its value is missing")
    func coalescedKeys() throws {
        let leftValues: [Double?] = [nil, -0.0, Double(bitPattern: 0x7ff8000000000042), .infinity]
        let rightValues: [Double?] = [7, 8, nil, -.infinity]
        for nativeLeft in [false, true] {
            for nativeRight in [false, true] where nativeLeft || nativeRight {
                let left = nativeLeft ? dense(leftValues, name: "x") : TypedColumn(name: "x", values: leftValues)
                let right = nativeRight ? dense(rightValues, name: "x") : TypedColumn(name: "x", values: rightValues)
                for count in [0, 1, 63, 64, 65, 129] {
                    let lhs: [Int?] = (0..<count).map { $0 % 3 == 0 ? nil : $0 % 4 }
                    let rhs: [Int?] = (0..<count).map { $0 % 5 == 0 ? nil : (3 - $0 % 4) }
                    let expected = (0..<count).map { row -> Double? in
                        if let index = lhs[row] { return leftValues[index] }
                        if let index = rhs[row] { return rightValues[index] }
                        return nil
                    }
                    let actual = try #require(left.gatheredJoinKey(leftIndices: lhs, rightColumn: right,
                        rightIndices: rhs, name: "key") as? TypedColumn<Double>)
                    try check(actual, expected, name: "key")
                }
            }
        }
    }

    private func dense(_ values: [Double?], name: String) -> TypedColumn<Double> {
        var builder = CSVCompactDoubleBuilder(capacity: values.count)
        for value in values { builder.append(value) }
        return builder.column(named: name)
    }

    private func fixture(_ row: Int) -> Double? {
        switch row % 6 {
        case 0: nil
        case 1: Double(bitPattern: 0x7ff8000000000042)
        case 2: -0.0
        case 3: .infinity
        case 4: -.infinity
        default: Double(row)
        }
    }

    private func check(_ column: TypedColumn<Double>, _ expected: [Double?], name: String) throws {
        let compact = try #require(column.compactNumericStorage)
        #expect(column.name == name && column.dtype == .float64)
        #expect(column.count == expected.count)
        #expect(column.nullCount == expected.filter { $0 == nil }.count)
        #expect(compact.nullCount == column.nullCount)
        for row in expected.indices { #expect(column[row]?.bitPattern == expected[row]?.bitPattern) }
    }
}
