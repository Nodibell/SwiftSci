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

    @Test func predictionOverflowReleasesCapacityAndAllowsReuse() async throws {
        let trainingColumns: [[Double]] = [[-1, 1]]
        let training = try PreparedNumericBatch(columnNames: ["x"], columns: trainingColumns)
        let model = MLPRegressor(hiddenLayerSizes: [], solver: .sgd,
            maxIter: 1, learningRate: 1, seed: 42, requestedDevice: .cpu)
        let fitted = try await model.fitStandardized(features: training, targets: [-2, 2])
        let budget = try MemoryBudget(limit: 1024 * 1024)
        let largeColumns: [[Double]] = [[1e308]]
        let large = try PreparedNumericBatch(columnNames: ["x"], columns: largeColumns)
        let input = try await fitted.prepare(MLPPredictionRequest(source: large), budget: budget)
        defer { withExtendedLifetime(input) {} }
        await #expect(throws: SwiftMLError.self) {
            try await fitted.predict(input, budget: budget)
        }
        #expect(await budget.reservedBytes == 0)
        #expect(await budget.queuedCount == 0)
        let ordinaryColumns: [[Double]] = [[0]]
        let ordinary = try PreparedNumericBatch(columnNames: ["x"], columns: ordinaryColumns)
        let reusable = try await fitted.prepare(MLPPredictionRequest(source: ordinary), budget: budget)
        let values = try await fitted.predict(reusable, budget: budget)
        #expect(values.count == 1)
        let allFinite: Bool = values.allSatisfy { $0.isFinite }
        #expect(allFinite)
        #expect(await budget.reservedBytes == 0)
        #expect(await budget.queuedCount == 0)
    }
}
