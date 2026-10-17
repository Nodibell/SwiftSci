import Testing
import SwiftML
import SwiftPreprocessing
import SwiftDataFrame

@Suite("Public standardized MLP exact admission")
struct StandardizedMLPAdmissionTests {
    @Test func completeQuotesAdmitOnceAndRejectBeforeWork() async throws {
        let training = try publicMLPTraining(), model = publicMLPModel()
        let fitted = try await model.fitStandardized(features: training.batch, targets: training.targets)
        var request = try MLPPredictionRequest(source: training.batch, selectingRows: [2, 1, 2, 0])
        try request.updateColumn(at: 0, rows: [0, 2, 2], values: [nil, 7, 8])
        let preparation = try fitted.preparationEstimate(for: request)
        let small = try MemoryBudget(limit: preparation.bytes - 1)
        await #expect(throws: MemoryAdmissionError.self) { try await fitted.prepare(request, budget: small) }
        #expect(await small.peak == 0)
        let exact = try MemoryBudget(limit: preparation.bytes)
        let input = try await fitted.prepare(request, budget: exact)
        #expect(await exact.peak == preparation.bytes)
        #expect(await exact.reservedBytes == 0)
        let prediction = try fitted.predictionEstimate(for: input)
        let smallPrediction = try MemoryBudget(limit: prediction.bytes - 1)
        await #expect(throws: MemoryAdmissionError.self) {
            try await fitted.predict(input, budget: smallPrediction)
        }
        #expect(await smallPrediction.peak == 0)
        let exactPrediction = try MemoryBudget(limit: prediction.bytes)
        #expect(try await fitted.predict(input, budget: exactPrediction).count == 4)
        #expect(await exactPrediction.peak == prediction.bytes)
        #expect(await exactPrediction.reservedBytes == 0)
    }
}
