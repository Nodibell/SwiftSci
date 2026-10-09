import CoreML
import Testing
import SwiftML
import SwiftPreprocessing

@Suite("Core ML compact matrix transport", .serialized)
struct CoreMLCompactMatrixTests {
    @Test(arguments: [(true, true), (true, false), (false, true), (false, false)], [7, 512])
    func stridedColumnsPreserveMappingAndOwnedResults(modes: (Bool, Bool), width: Int) async throws {
        let (float32, asynchronous) = modes
        let rows = 513
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [rows, width], float32: float32)) { url in
            let names = (0..<width).map { "x\($0)" }
            let predictor = try CoreMLPredictor(compiledModelURL: url, inputColumns: names,
                outputName: "result", computeUnits: .cpuOnly, asynchronousMatrix: asynchronous)
            let columns = (0..<width).map { c in
                (0..<rows).map { r in Double((r * 13 + c * 17) % 101 - 50) / 8 }
            }
            let raw = try PreparedNumericBatch(columnNames: Array(names.reversed()), columns: Array(columns.reversed()))
            let selection = (0..<rows).map { rows - 1 - ($0 / 2) }
            let input = try raw.selectingRows(selection)
            let budget = try MemoryBudget(limit: 16 * 1_048_576)
            let result = try await predictor.predict(input, budget: budget, workspaceBytes: 128)
            #expect(result.values.originalRowIndices == selection)
            #expect(result.values.columnNames == (0..<width).map { "result[\($0)]" })
            for col in 0..<width {
                let expected = selection.map { max(0, columns[col][$0]) }
                #expect(result.values.columnValues(at: col) == expected.map(Optional.some))
            }
            let bytes = rows * width * (float32 ? 4 : 8)
            #expect(result.inputPackingBytes == bytes)
            #expect(result.outputCopyBytes == rows * width * 8)
            #expect(await budget.peak == input.payloadByteCount + rows * width * 8 + bytes * 2 + 128)
            var changed = input
            try changed.updateColumn(at: 0, rows: [0, 256, 512], values: [0, 0, 0])
            _ = try await predictor.predict(changed, budget: budget, workspaceBytes: 128)
            #expect(result.values[256, width - 1] == max(0, columns[width - 1][selection[256]]))
            #expect(input[256, 0] == columns[width - 1][selection[256]])
            #expect(await budget.reservedBytes == 0)
        }
    }

    @Test(arguments: [false, true])
    func lateInvalidColumnDrainsReservationAndAllowsReuse(asynchronous: Bool) async throws {
        try await withCompiledCoreML(coreMLReLUArtifact(shape: [513, 3])) { url in
            let predictor = try CoreMLPredictor(compiledModelURL: url, inputColumns: ["x", "y", "z"],
                outputName: "result", computeUnits: .cpuOnly, asynchronousMatrix: asynchronous)
            let columns = [[Double]](repeating: [Double](repeating: 1, count: 513), count: 3)
            let input = try PreparedNumericBatch(columnNames: ["x", "y", "z"], columns: columns)
            let budget = try MemoryBudget(limit: 16 * 1_048_576)
            for row in [255, 256, 512] {
                for invalid: Double? in [nil, .nan, .infinity, .greatestFiniteMagnitude] {
                    var bad = input
                    try bad.updateColumn(at: 2, rows: [row], values: [invalid])
                    await #expect(throws: SwiftMLError.self) {
                        try await predictor.predict(bad, budget: budget, workspaceBytes: 0)
                    }
                    #expect(await budget.reservedBytes == 0)
                }
            }
            let valid = try await predictor.predict(input, budget: budget, workspaceBytes: 0)
            #expect(valid.values.columnValues(at: 2) == [Double?](repeating: 1, count: 513))
            #expect(await budget.reservedBytes == 0)
        }
    }
}
