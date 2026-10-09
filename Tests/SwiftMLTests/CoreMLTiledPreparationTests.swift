import CoreML
import Testing
import SwiftML
import SwiftPreprocessing

@Suite("Tiled fitted model preparation", .serialized)
struct CoreMLTiledPreparationTests {
    @Test func partialFinalTilePreservesResultsRowsAndAdmission() async throws {
        let width = 64
        let rowCount = 83
        var trainingColumns: [[Double]] = []
        trainingColumns.reserveCapacity(width)
        for column in 0..<width {
            let first: Double = Double(column)
            let second: Double = Double(column + 1)
            let third: Double = Double(column + 3)
            let values: [Double] = [first, second, third]
            trainingColumns.append(values)
        }
        let names = (0..<width).map { "x\($0)" }
        let training = try PreparedNumericBatch(columnNames: names, columns: trainingColumns)
        let source = try selectedInput(names: names, rows: rowCount)
        let plan = try StandardPreprocessingPlan(training: training)
        var imputer = Imputer()
        try imputer.fit(training)
        var scaler = StandardScaler()
        try scaler.fit(imputer.transform(training))
        let scaled = try scaler.transform(imputer.transform(source))
        let inputBudget = try MemoryBudget(limit: 1_048_576)
        let poolBudget = try MemoryBudget(limit: 2_097_152)
        let config = try CoreMLMatrixPool.Configuration(retainedBytes: 1_048_576, requestBytes: 131_072)
        let artifact = coreMLReLUArtifact(shape: [rowCount, width])
        try await withCompiledCoreML(artifact) { url in
            try await CoreMLMatrixPool.withPool(compiledModelURL: url, inputColumns: names,
                outputName: "result", computeUnits: .cpuOnly, budget: poolBudget, configuration: config) { pool in
                let preparation = try await pool.inputPreparation()
                let direct = try await preparation.prepare(source, preprocessing: plan, budget: inputBudget)
                let workspace = try plan.workspaceAllowance(for: source)
                #expect(await inputBudget.peak == direct.reservedBytes + workspace.bytes)
                #expect(await inputBudget.reservedBytes == direct.reservedBytes)
                let reference = try await pool.prepare(scaled, budget: inputBudget)
                let expected = try await pool.predict(reference)
                let actual = try await pool.predict(direct)
                let actualValues = try actual.values.matrix().values
                let expectedValues = try expected.values.matrix().values
                #expect(actualValues == expectedValues)
                #expect(actual.values.originalRowIndices == source.originalRowIndices)
                #expect(actual.values.originalRowIndices.first == actual.values.originalRowIndices.last)
            }
        }
        await waitForPreparedRelease(inputBudget)
        #expect(await poolBudget.reservedBytes == 0)
    }

    private func selectedInput(names: [String], rows: Int) throws -> PreparedNumericBatch {
        var columns = [[Double]]()
        for column in names.indices {
            var values = [Double]()
            for row in 0..<rows {
                let value = Double((row * 3 + column) % 23)
                values.append((row + column) % 11 == 0 ? .nan : value)
            }
            columns.append(values)
        }
        var input = try PreparedNumericBatch(columnNames: names, columns: columns)
        try input.updateColumn(at: 0, rows: [7], values: [nil])
        var selection = Array((0..<rows).reversed())
        selection[rows - 1] = rows - 1
        return try input.selectingRows(selection)
    }
}
