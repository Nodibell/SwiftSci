import Foundation
import Testing
@testable import SwiftML
import SwiftPreprocessing

@Suite("Batch pipeline fitted-column boundary")
struct CoreMLBatchSessionBoundaryTests {
    @Test func mismatchedColumnsRejectBeforeModelLoading() async throws {
        let (_, plan) = try pipelineFixture()
        let config = try pipelineConfiguration()
        let lifetime = try MemoryBudget(limit: config.sessionBytes)
        let work = try MemoryBudget(limit: config.windowBytes)
        let missing = URL(fileURLWithPath: "/nonexistent-model")
        let expected = SwiftMLError.invalidParameter("Pipeline columns differ from the fitted plan")
        await #expect(throws: expected) {
            try await CoreMLBatchPipeline.withSession(compiledModelURL: missing,
                inputColumns: ["wrong"], outputName: "result", preprocessing: plan,
                lifetimeBudget: lifetime, workBudget: work, configuration: config) { _ in }
        }
        await #expect(throws: expected) {
            try await CoreMLBatchPipeline.run(compiledModelURL: missing,
                inputColumns: ["wrong"], outputName: "result", preprocessing: plan,
                budget: lifetime, configuration: config,
                load: { _ in nil }, consume: { _, _ in })
        }
        #expect(await lifetime.reservedBytes == 0)
        #expect(await work.reservedBytes == 0)
    }
}
