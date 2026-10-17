import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

@Suite("Shared immutable CPU predictors", .serialized)
struct MLPCPUPredictorConcurrencyTests {
    @Test func concurrentCallsOwnTheirWorkspaces() async throws {
        let c = try await preparedMLPTestContext(width: 128)
        let view = try IndexedNumericBatch(c.source, selectingRows: Array((0..<1031).reversed()))
        let setup = try MemoryBudget(limit: 32 * 1024 * 1024)
        let input = try await PreparedMLPInput.prepare(features: view, preprocessing: c.plan, budget: setup)
        let expected = try c.fitted.predictExternallyAdmitted(features: input)
        let estimate = try c.fitted.predictionEstimate(for: input)
        let budget = try MemoryBudget(limit: estimate.bytes * 2)
        let results = try await withThrowingTaskGroup(of: [Double].self) { group in
            for _ in 0..<8 { group.addTask { try await c.fitted.predict(features: input, budget: budget) } }
            var values: [[Double]] = []
            for try await value in group { values.append(value) }
            return values
        }
        #expect(results.count == 8)
        for value in results { #expect(value.map(\.bitPattern) == expected.map(\.bitPattern)) }
        #expect(await budget.peak <= estimate.bytes * 2)
        #expect(await budget.reservedBytes == 0)
        #expect(await budget.queuedCount == 0)
    }
}
