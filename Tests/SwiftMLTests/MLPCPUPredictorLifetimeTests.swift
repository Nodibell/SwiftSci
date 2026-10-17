import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

@Suite("CPU predictor ownership", .serialized)
struct MLPCPUPredictorLifetimeTests {
    @Test func snapshotDoesNotRetainItsModelActor() async throws {
        let rows = matrixMLPFeatures(8, width: 7)
        let source = try matrixMLPBatch(rows, width: 7)
        let plan = try StandardPreprocessingPlan(training: source)
        var model: MLPRegressor? = MLPRegressor(hiddenLayerSizes: [9, 5], maxIter: 1, seed: 42, requestedDevice: .cpu)
        try await model!.fit(features: rows, targets: rows.map { $0[0] })
        var fitted: MLPCPUPredictor? = try await model!.makeCPUPredictor()
        weak var actorWitness = model
        weak var fittedWitness = fitted
        let view = try IndexedNumericBatch(source, selectingRows: [7, 0, 7, 2])
        let budget = try MemoryBudget(limit: 8 * 1024 * 1024)
        let input = try await PreparedMLPInput.prepare(features: view, preprocessing: plan, budget: budget)
        let expected = try await model!.predictSourceOrdered(features: view, preprocessing: plan)
        model = nil
        #expect(actorWitness == nil)
        let actual = try await fitted!.predict(features: input, budget: budget)
        #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
        fitted = nil
        #expect(fittedWitness == nil)
        #expect(await budget.reservedBytes == 0)
    }

    @Test func unfittedCaptureThrows() async throws {
        let model = MLPRegressor(requestedDevice: .cpu)
        await #expect(throws: SwiftMLError.modelNotFitted) { try await model.makeCPUPredictor() }
    }
}
