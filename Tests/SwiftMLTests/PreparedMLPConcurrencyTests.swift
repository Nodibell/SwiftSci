import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

@Suite("Shared prepared MLP input", .serialized)
struct PreparedMLPConcurrencyTests {
    @Test func concurrentReuseFitsOneWholePredictionAtATime() async throws {
        let c = try await preparedMLPTestContext()
        let view = try IndexedNumericBatch(c.source, selectingRows: Array((0..<65).reversed()))
        let setup = try MemoryBudget(limit: 8 * 1024 * 1024)
        let input = try await PreparedMLPInput.prepare(features: view, preprocessing: c.plan, budget: setup)
        let expected = try await c.model.predictSourceOrdered(features: view, preprocessing: c.plan)
        let estimate = try c.fitted.predictionEstimate(for: input)
        let budget = try MemoryBudget(limit: estimate.bytes)
        let results = try await withThrowingTaskGroup(of: [Double].self) { group in
            for _ in 0..<4 { group.addTask { try await c.model.predict(features: input, budget: budget) } }
            var results: [[Double]] = []
            for try await result in group { results.append(result) }
            return results
        }
        #expect(results.count == 4)
        for result in results { #expect(result.map(\.bitPattern) == expected.map(\.bitPattern)) }
        #expect(await budget.peak == estimate.bytes)
        #expect(await budget.reservedBytes == 0)
        #expect(await budget.queuedCount == 0)
        #expect(await setup.reservedBytes == 0)
    }

    @Test func schemaMismatchIsRejectedBeforeAdmission() async throws {
        let c = try await preparedMLPTestContext()
        let other = try await preparedMLPTestContext(width: 8)
        let view = try IndexedNumericBatch(c.source, selectingRows: [3, 0, 1])
        let budget = try MemoryBudget(limit: 8 * 1024 * 1024)
        await #expect(throws: SwiftMLError.self) {
            try await PreparedMLPInput.prepare(features: view, preprocessing: other.plan, budget: budget)
        }
        #expect(await budget.peak == 0)
        #expect(await budget.reservedBytes == 0)
    }
    @Test func callerManagedGroupUsesOneReservationForRepeatedPredictions() async throws {
        let c = try await preparedMLPTestContext()
        let view = try IndexedNumericBatch(c.source, selectingRows: Array((0..<65).reversed()))
        let budget = try MemoryBudget(limit: 8 * 1024 * 1024)
        let input = try await PreparedMLPInput.prepare(features: view, preprocessing: c.plan, budget: budget)
        let expected = try await c.model.predict(features: input, budget: budget)
        let estimate = try c.fitted.predictionEstimate(for: input)
        let reservation = try await budget.acquire(estimate)
        do {
            for _ in 0..<4 {
                let result = try c.fitted.predictExternallyAdmitted(features: input)
                #expect(result.map(\.bitPattern) == expected.map(\.bitPattern))
                #expect(await budget.reservedBytes == estimate.bytes)
            }
            await reservation.finish()
        } catch {
            await reservation.finish()
            throw error
        }
        #expect(await budget.reservedBytes == 0)
    }

}
