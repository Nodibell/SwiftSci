import Testing
import SwiftML
import SwiftPreprocessing
import SwiftDataFrame

@Suite("Public standardized MLP numerical boundaries")
struct StandardizedMLPBoundaryTests {
    @Test func failedTrainingAndIncomingValidationReleaseCapacity() async throws {
        let training = try publicMLPTraining(), model = publicMLPModel()
        let fitted = try await model.fitStandardized(features: training.batch, targets: training.targets)
        await #expect(throws: SwiftMLError.self) {
            try await model.fitStandardized(features: training.batch, targets: [.infinity])
        }
        await #expect(throws: SwiftMLError.self) {
            try await model.fitStandardized(features: training.batch, targets: training.targets,
                imputationStrategy: .constant(.nan))
        }
        let budget = try MemoryBudget(limit: 16 * 1024 * 1024)
        var source = training.batch
        try source.updateColumn(at: 0, rows: [1], values: [.infinity])
        await #expect(throws: SwiftMLError.self) {
            try await fitted.prepare(MLPPredictionRequest(source: source), budget: budget)
        }
        let selected = try MLPPredictionRequest(source: source, selectingRows: [0, 2])
        let input = try await fitted.prepare(selected, budget: budget)
        #expect(try await fitted.predict(input, budget: budget).allSatisfy(\.isFinite))
        let names: [String] = Array(training.batch.columnNames.reversed())
        let columns = [[Double]](repeating: [], count: names.count)
        let empty = try PreparedNumericBatch(columnNames: names, columns: columns)
        #expect(throws: SwiftMLError.self) {
            try fitted.preparationEstimate(for: MLPPredictionRequest(source: empty))
        }
        #expect(await budget.reservedBytes == 0)
        #expect(await budget.queuedCount == 0)
    }
}
