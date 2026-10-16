import Testing
@testable import SwiftDataFrame

@Suite("Compact row gather semantics")
struct CompactRowGatherTests {
    @Test(arguments: [0, 1, 63, 64, 65, 130, 1024])
    func nullablePermutationAndSnapshots(rows: Int) throws {
        let special = Double(bitPattern: 0x7ff8_0000_0000_0123)
        let source: [Double?] = (0..<rows).map { row in
            if row % 17 == 0 { return nil }
            if row % 19 == 0 { return special }
            if row % 23 == 0 { return -0.0 }
            return Double(row) / 8
        }
        let frame = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: source)])
        let batch = try frame.prepareNumericBatch(["x", "x"])
        let indices = rows == 0 ? [] : (0..<(rows + 67)).map { ($0 * 31) % rows }
        var selected = try batch.selectingRows(indices)
        let expected = indices.map { source[$0] }
        #expect(matches(selected.columnValues(at: 0), expected))
        #expect(matches(selected.columnValues(at: 1), expected))
        #expect(selected.nullCount(inColumn: 0) == expected.filter { $0 == nil }.count)
        #expect(selected.originalRowIndices == indices)
        let reversed = Array(indices.indices.reversed())
        let chained = try selected.selectingRows(reversed)
        #expect(chained.originalRowIndices == indices.reversed())
        #expect(matches(chained.columnValues(at: 0), Array(expected.reversed())))
        let snapshot = selected
        if !indices.isEmpty { try selected.updateColumn(at: 0, rows: [0], values: [1234]) }
        #expect(matches(snapshot.columnValues(at: 0), expected))
        #expect(matches(batch.columnValues(at: 0), source))
        #expect(matches(selected.columnValues(at: 1), expected))
    }

    @Test(arguments: [1, 7, 64, 129])
    func wideDenseAndNullableGathers(width: Int) throws {
        let rows = 2048
        let names = (0..<width).map { "x\($0)" }
        let columns: [[Double]] = (0..<width).map { column -> [Double] in
            (0..<rows).map { row -> Double in Double(row + column * rows) }
        }
        var batch = try PreparedNumericBatch(columnNames: names, columns: columns)
        for column in stride(from: 0, to: width, by: 3) {
            try batch.updateColumn(at: column, rows: [0, 63, 64, 2047], values: [nil, nil, nil, nil])
        }
        let indices = (0..<rows).map { ($0 * 7919) % rows } + [64, 0, 64]
        let selected = try batch.selectingRows(indices)
        #expect(selected.originalRowIndices == indices)
        for column in 0..<width {
            let source = batch.columnValues(at: column)
            #expect(matches(selected.columnValues(at: column), indices.map { source[$0] }))
            #expect(selected.nullCount(inColumn: column) == indices.filter { source[$0] == nil }.count)
        }
    }

    @Test func allValidSelectionDropsBitmapAndBoundsRemainChecked() throws {
        let frame = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: [nil, 1, nil, 2])])
        let batch = try frame.prepareNumericBatch(["x"])
        let valid = try batch.selectingRows([3, 1, 1])
        #expect(valid.columnValues(at: 0) == [2, 1, 1])
        #expect(valid.payloadByteCount == 3 * MemoryLayout<Double>.stride)
        let missing = try batch.selectingRows([0, 2, 0])
        #expect(missing.nullCount(inColumn: 0) == 3)
        #expect(missing.columnValues(at: 0).allSatisfy { $0 == nil })
        #expect(throws: (any Error).self) { try batch.selectingRows([-1]) }
        #expect(throws: (any Error).self) { try batch.selectingRows([4]) }
        let empty = try batch.selectingRows([])
        #expect(empty.rowCount == 0 && empty.payloadByteCount == 0)
    }

    @Test(arguments: [1, 2, 63, 64, 65])
    func singleValidityWordBoundary(count: Int) throws {
        var values: [Double?] = (0..<130).map { $0 % 17 == 0 ? nil : Double($0) }
        values[2] = Double(bitPattern: 0x7ff8_0000_0000_0456)
        values[3] = -0.0
        let frame = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: values)])
        let batch = try frame.prepareNumericBatch(["x"])
        var indices = (0..<count).map { ($0 * 31 + 2) % values.count }
        if count > 1 { indices[1] = 3 }
        if count > 2 { indices[2] = 2 }
        let selected = try batch.selectingRows(indices)
        let expected = indices.map { values[$0] }
        let missing = expected.filter { $0 == nil }.count
        #expect(matches(selected.columnValues(at: 0), expected))
        #expect(selected.nullCount(inColumn: 0) == missing)
        #expect(selected.payloadByteCount == count * 8 + (missing == 0 ? 0 : ((count + 63) / 64) * 8))
        #expect(matches(batch.columnValues(at: 0), values))
    }

    private func matches(_ actual: [Double?], _ expected: [Double?]) -> Bool {
        actual.count == expected.count && zip(actual, expected).allSatisfy { a, b in
            switch (a, b) {
            case (nil, nil): return true
            case let (a?, b?): return a.bitPattern == b.bitPattern
            default: return false
            }
        }
    }
}
