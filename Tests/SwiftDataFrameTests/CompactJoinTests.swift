import Testing
@testable import SwiftDataFrame

@Suite("Compact join integration")
struct CompactJoinTests {
    @Test("Every join kind preserves compact payloads, coalesced keys and exact native integers")
    func publicJoins() throws {
        for side in 0..<3 {
            let nativeLeft = side != 1
            let nativeRight = side != 0
            let left = try frame(right: false, native: nativeLeft)
            let right = try frame(right: true, native: nativeRight)
            let referenceLeft = try frame(right: false, native: false)
            let referenceRight = try frame(right: true, native: false)
            for kind in [JoinKind.inner, .left, .right, .outer] {
                let actual = try left.join(right, on: "key", how: kind)
                let expected = try referenceLeft.join(referenceRight, on: "key", how: kind)
                #expect(actual.columnNames == expected.columnNames)
                #expect(actual.rowCount == expected.rowCount)
                var denseNames = ["key"]
                if nativeLeft { denseNames.append("value_x") }
                if nativeRight { denseNames.append("value_y") }
                for name in ["key", "value_x", "value_y"] {
                    let column = try #require(actual[column: name, as: Double.self])
                    let reference = try #require(expected[column: name, as: Double.self])
                    #expect(column.nullCount == reference.nullCount)
                    compare(column.values, reference.values)
                    if denseNames.contains(name) { #expect(column.compactNumericStorage != nil) }
                }
                for name in ["identifier_x", "identifier_y"] {
                    let column = try #require(actual[column: name, as: Int64.self])
                    let reference = try #require(expected[column: name, as: Int64.self])
                    #expect(column.values == reference.values)
                    #expect(column.nullCount == reference.nullCount)
                }
                let packed = try actual.prepareNumericMatrix(["value_x", "value_y"])
                let reference = try expected.prepareNumericMatrix(["value_x", "value_y"])
                compare(packed.values.map(Optional.some), reference.values.map(Optional.some))
                let storage = try #require(actual[column: "key", as: Double.self]?.compactNumericStorage)
                var batch = try actual.prepareNumericBatch(["key"])
                #expect(storage.values.withUnsafeBufferPointer { lhs in
                    batch.columns[0].values.withUnsafeBufferPointer { rhs in lhs.baseAddress == rhs.baseAddress }
                })
                let first = actual[column: "key", as: Double.self]![0]?.bitPattern
                try batch.updateColumn(at: 0, rows: [0], values: [999])
                #expect(actual[column: "key", as: Double.self]![0]?.bitPattern == first)
                #expect(batch[0, 0] == 999)
            }
        }
    }

    @Test func floatingKeyEqualityMatchesOptionalInputs() throws {
        let leftKeys: [Double?] = [-0.0, .nan, nil, 1]
        let rightKeys: [Double?] = [0.0, .nan, nil, 1]
        let identifiers = TypedColumn<Int64>(name: "id", values: [1, 2, 3, 4])
        for compactLeft in [false, true] {
            for compactRight in [false, true] {
                let lhs = compactLeft ? dense(leftKeys, name: "key")
                    : TypedColumn<Double>(name: "key", values: leftKeys)
                let rhs = compactRight ? dense(rightKeys, name: "key")
                    : TypedColumn<Double>(name: "key", values: rightKeys)
                let left = try DataFrame(columns: [lhs, identifiers])
                let right = try DataFrame(columns: [rhs, identifiers])
                let result = try left.join(right, on: "key")
                #expect(result.rowCount == 2)
                #expect(result[column: "id_x", as: Int64.self]?.values == [1, 4])
                #expect(result[column: "id_y", as: Int64.self]?.values == [1, 4])
                let keys = try #require(result[column: "key", as: Double.self])
                #expect(keys[0]?.bitPattern == (-0.0).bitPattern)
                #expect(keys[1] == 1)
            }
        }
    }

    private func frame(right: Bool, native: Bool) throws -> DataFrame {
        let count = right ? 133 : 130
        let keyOffset: Int = right ? 7 : 0
        let keys: [Double?] = (0..<count).map { (row: Int) -> Double? in
            if row % 19 == 0 { return nil }
            let key: Int = row % 17 + keyOffset
            return Double(key)
        }
        let values: [Double?] = (0..<count).map { (row: Int) -> Double? in
            if row % 11 == 0 { return nil }
            if row % 11 == 1 { return Double(bitPattern: 0x7ff8000000000042) }
            if row % 11 == 2 { return -0.0 }
            return Double(row)
        }
        let key = native ? dense(keys, name: "key") : TypedColumn(name: "key", values: keys)
        let value = native ? dense(values, name: "value") : TypedColumn(name: "value", values: values)
        let identifiers = (0..<count).map { Int64($0) + 9_007_199_254_740_993 }
        return try DataFrame(columns: [key, value, TypedColumn(name: "identifier", values: identifiers)])
    }

    private func dense(_ values: [Double?], name: String) -> TypedColumn<Double> {
        var builder = CSVCompactDoubleBuilder(capacity: values.count)
        for value in values { builder.append(value) }
        return builder.column(named: name)
    }

    private func compare(_ actual: [Double?], _ expected: [Double?]) {
        #expect(actual.count == expected.count)
        for (actual, expected) in zip(actual, expected) { #expect(actual?.bitPattern == expected?.bitPattern) }
    }
}
