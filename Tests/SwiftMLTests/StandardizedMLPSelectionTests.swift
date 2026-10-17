import Testing
import SwiftML
import SwiftPreprocessing
import SwiftDataFrame

@Suite("Public MLP selections and deferred edits")
struct StandardizedMLPSelectionTests {
    @Test func duplicateOccurrencesAndRequestSnapshots() async throws {
        let training = try publicMLPTraining()
        let fitted = try await publicMLPModel().fitStandardized(features: training.batch, targets: training.targets)
        let reference = try await publicMLPReference(training.batch, targets: training.targets)
        var source = try matrixMLPBatch(matrixMLPFeatures(65, width: 7), width: 7)
        let positions = [64, 2, 2, 0] + Array(0..<33)
        var request = try MLPPredictionRequest(source: source, selectingRows: positions)
        let oldRequest = request
        try request.updateColumn(at: 0, rows: [0, 2, 2], values: [99, nil, 12])
        try request.updateColumn(at: 0, rows: [1], values: [.nan])
        var expectedBatch = try source.selectingRows(positions)
        try expectedBatch.updateColumn(at: 0, rows: [0, 2, 2], values: [99, nil, 12])
        try expectedBatch.updateColumn(at: 0, rows: [1], values: [.nan])
        try source.updateColumn(at: 0, rows: [64], values: [-999])
        let budget = try MemoryBudget(limit: 16 * 1024 * 1024)
        let input = try await fitted.prepare(request, budget: budget)
        let actual = try await fitted.predict(input, budget: budget)
        let expected = try await reference.predict(expectedBatch)
        #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
        let oldInput = try await fitted.prepare(oldRequest, budget: budget)
        #expect(try await fitted.predict(oldInput, budget: budget) != actual)
        #expect(await budget.reservedBytes == 0)
    }
    @Test func invalidEditLeavesRequestUnchanged() throws {
        let batch = try matrixMLPBatch(matrixMLPFeatures(3, width: 7), width: 7)
        var request = try MLPPredictionRequest(source: batch, selectingRows: [2, 0, 2])
        #expect(throws: SwiftMLError.self) { try request.updateColumn(at: 0, rows: [3], values: [1]) }
        #expect(throws: SwiftMLError.self) { try request.updateColumn(at: 7, rows: [], values: []) }
        #expect(throws: SwiftMLError.self) { try MLPPredictionRequest(source: batch, selectingRows: [-1]) }
        #expect(request.rowCount == 3)
    }
}
