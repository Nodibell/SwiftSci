import Testing
import SwiftML
import SwiftPreprocessing

@Suite("Public standardized MLP ownership")
struct StandardizedMLPOwnershipTests {
    @Test func inputDoesNotRetainWeightsOwnerOrTrainingActor() async throws {
        let training = try publicMLPTraining()
        var model: MLPRegressor? = publicMLPModel()
        weak var actor = model
        var fitted: FittedMLPPredictor? = try await model!.fitStandardized(features: training.batch, targets: training.targets)
        weak var owner = fitted
        let budget = try MemoryBudget(limit: 16 * 1024 * 1024)
        var input: MLPPredictionInput? = try await fitted!.prepare(MLPPredictionRequest(source: training.batch), budget: budget)
        weak var data = input
        model = nil
        #expect(actor == nil)
        #expect(try await fitted!.predict(input!, budget: budget).count == training.batch.rowCount)
        fitted = nil
        #expect(owner == nil && input != nil)
        input = nil
        #expect(data == nil)
        #expect(await budget.reservedBytes == 0)
    }
}
