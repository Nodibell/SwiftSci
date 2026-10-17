import Testing
import SwiftML
import SwiftPreprocessing
import SwiftDataFrame

@Suite("Public MLP fitted context identity")
struct StandardizedMLPIdentityTests {
    @Test func refitsAndForeignInputs() async throws {
        let training = try publicMLPTraining(), model = publicMLPModel()
        let first = try await model.fitStandardized(features: training.batch, targets: training.targets)
        let budget = try MemoryBudget(limit: 16 * 1024 * 1024)
        let input = try await first.prepare(MLPPredictionRequest(source: training.batch), budget: budget)
        let before = try await first.predict(input, budget: budget)
        let targets: [Double] = training.targets.map { -$0 + 10 }
        let second = try await model.fitStandardized(features: training.batch, targets: targets)
        #expect(try await first.predict(input, budget: budget) == before)
        #expect(throws: SwiftMLError.self) { try second.predictionEstimate(for: input) }
        let tiny = try MemoryBudget(limit: 1)
        await #expect(throws: SwiftMLError.self) { try await second.predict(input, budget: tiny) }
        #expect(await tiny.queuedCount == 0)
        #expect(await tiny.reservedBytes == 0)
    }
    @Test func concurrentFitsKeepInvocationStatisticsAndWeights() async throws {
        let a = try publicMLPTraining()
        var b = a.batch
        try b.updateColumn(at: 0, rows: Array(0..<18), values: Array(repeating: 1000, count: 18))
        let targets: [Double] = a.targets.map { -$0 + 10 }
        let batchB = b
        let model = publicMLPModel()
        async let left = model.fitStandardized(features: a.batch, targets: a.targets)
        async let right = model.fitStandardized(features: batchB, targets: targets)
        let (first, second) = try await (left, right)
        let ra = try await publicMLPReference(a.batch, targets: a.targets)
        let rb = try await publicMLPReference(batchB, targets: targets)
        let budget = try MemoryBudget(limit: 16 * 1024 * 1024)
        let request = MLPPredictionRequest(source: a.batch)
        let ia = try await first.prepare(request, budget: budget)
        let ib = try await second.prepare(request, budget: budget)
        let actualA = try await first.predict(ia, budget: budget)
        let actualB = try await second.predict(ib, budget: budget)
        let expectedA = try await ra.predict(a.batch)
        let expectedB = try await rb.predict(a.batch)
        #expect(actualA.map(\.bitPattern) == expectedA.map(\.bitPattern))
        #expect(actualB.map(\.bitPattern) == expectedB.map(\.bitPattern))
    }
}
