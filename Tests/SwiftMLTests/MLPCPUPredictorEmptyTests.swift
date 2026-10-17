import Testing
import SwiftDataFrame
import SwiftPreprocessing
@testable import SwiftML

@Suite("Empty CPU snapshot inputs", .serialized)
struct MLPCPUPredictorEmptyTests {
    @Test func emptyInputStillRequiresMatchingModelWidth() async throws {
        let c = try await preparedMLPTestContext()
        let other = try await preparedMLPTestContext(width: 8)
        let view = try IndexedNumericBatch(other.source, selectingRows: [])
        let budget = try MemoryBudget(limit: 8 * 1024 * 1024)
        let input = try await PreparedMLPInput.prepare(features: view, preprocessing: other.plan, budget: budget)
        #expect(throws: SwiftMLError.dimensionMismatch(expected: 7, got: 8)) {
            try c.fitted.predictExternallyAdmitted(features: input)
        }
        await #expect(throws: SwiftMLError.dimensionMismatch(expected: 7, got: 8)) {
            try await c.fitted.predict(features: input, budget: budget)
        }
        #expect(await budget.reservedBytes == 0)
    }
}
