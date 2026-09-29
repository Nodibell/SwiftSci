import Testing
@testable import SwiftDataFrame

@Suite("Compact batch mutation")
struct CompactMutationTests {
    private func makeBatch() throws -> PreparedNumericBatch {
        try DataFrame(columns: [TypedColumn<Double>(name: "x", values: (0..<130).map(Double.init)),
                                TypedColumn<Double>(name: "y", values: [Double](repeating: 7, count: 130))]).prepareNumericBatch(["x", "y"])
    }

    @Test func denseInitializationPreservesValuesAndChecksShape() throws {
        let values = [1.0, Double.nan, 3]
        var batch = try PreparedNumericBatch(columnNames: ["x"], columns: [values])
        #expect(batch.payloadByteCount == 24 && batch.nullCount(inColumn: 0) == 0)
        try batch.updateColumn(at: 0, rows: [1], values: [nil])
        #expect(values[1].isNaN && batch[1,0] == nil)
        #expect(throws: (any Error).self) { try PreparedNumericBatch(columnNames: ["x"], columns: []) }
        #expect(throws: (any Error).self) { try PreparedNumericBatch(columnNames: ["x","y"], columns: [[1], [2,3]]) }
    }

    @Test func snapshotsValidityAndRepeatedIndices() throws {
        var batch = try makeBatch()
        let snapshot = batch
        let matrix = try batch.matrix(order: .columnMajor)
        try batch.updateColumn(at: 0, rows: [0, 63, 64, 129, 64, 64], values: [nil, nil, .nan, nil, nil, 42])
        #expect(batch.nullCount(inColumn: 0) == 3)
        #expect(batch[64, 0] == 42 && batch[63, 0] == nil && batch[129, 0] == nil)
        #expect(snapshot[64, 0] == 64 && snapshot.nullCount(inColumn: 0) == 0)
        #expect(matrix[64, 0] == 64)
        #expect(batch.columnValues(at: 1) == snapshot.columnValues(at: 1))
        let withNulls = batch
        try batch.updateColumn(at: 0, rows: [0, 63, 129], values: [.nan, 63, 129])
        #expect(batch[0,0]?.isNaN == true && batch.nullCount(inColumn: 0) == 0)
        #expect(batch.payloadByteCount == 2 * 130 * 8)
        #expect(withNulls[0,0] == nil && withNulls.nullCount(inColumn: 0) == 3)
    }

    @Test func validationIsAtomic() throws {
        var batch = try makeBatch()
        let before = batch.columnValues(at: 0)
        #expect(throws: (any Error).self) { try batch.updateColumn(at: 0, rows: [0, 130], values: [nil, nil]) }
        #expect(throws: (any Error).self) { try batch.updateColumn(at: 0, rows: [-1], values: [nil]) }
        #expect(throws: (any Error).self) { try batch.updateColumn(at: 0, rows: [0], values: []) }
        #expect(throws: (any Error).self) { try batch.updateColumn(at: 2, rows: [], values: []) }
        #expect(batch.columnValues(at: 0) == before && batch.nullCount(inColumn: 0) == 0)
        try batch.updateColumn(at: 0, rows: [], values: [])
        #expect(batch.columnValues(at: 0) == before)
    }

    @Test func singleColumnExportRetainsOldAllocation() throws {
        var batch = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: [1, 2, 3])]).prepareNumericBatch(["x"])
        let matrix = try batch.matrix()
        try batch.updateColumn(at: 0, rows: [1], values: [nil])
        #expect(matrix.values == [1, 2, 3])
        #expect(try batch.matrix().values[1].isNaN)
        #expect(batch.originalRowIndices == [0, 1, 2])
    }

    @Test func deterministicScatterMatchesOptionalReference() throws {
        var batch = try makeBatch()
        var reference = (0..<130).map { Optional(Double($0)) }
        for pass in 0..<40 {
            let indices = (0..<83).map { ($0 * 37 + pass * 19) % 130 }
            let replacements: [Double?] = indices.enumerated().map { i, row in
                (i + pass) % 5 == 0 ? nil : Double(row - pass)
            }
            try batch.updateColumn(at: 0, rows: indices, values: replacements)
            for (i, row) in indices.enumerated() { reference[row] = replacements[i] }
            #expect(batch.columnValues(at: 0) == reference)
            #expect(batch.nullCount(inColumn: 0) == reference.filter { $0 == nil }.count)
        }
    }

    @Test func separateCopiesCanMutateConcurrently() async throws {
        let original = try makeBatch()
        let nullCounts = try await withThrowingTaskGroup(of: Int.self) { group in
            for column in 0..<2 {
                group.addTask {
                    var copy = original
                    try copy.updateColumn(at: column, rows: [0, 63, 64], values: [nil, nil, nil])
                    return copy.nullCount(inColumn: column)
                }
            }
            var counts = [Int]()
            for try await count in group { counts.append(count) }
            return counts
        }
        #expect(nullCounts == [3,3])
        #expect(original.nullCount(inColumn: 0) == 0 && original.nullCount(inColumn: 1) == 0)
    }
}
