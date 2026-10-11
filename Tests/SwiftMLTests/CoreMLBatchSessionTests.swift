import CoreML
import Foundation
import Testing
@testable import SwiftML
import SwiftPreprocessing

@Suite("Retained Core ML batch sessions", .serialized)
struct CoreMLBatchSessionTests {
    @Test func sequentialRunsReleaseWorkspaceAndEscapedSessionRejects() async throws {
        let (source, plan) = try pipelineFixture()
        let config = try pipelineConfiguration()
        let lifetime = try MemoryBudget(limit: config.sessionBytes)
        let work = try MemoryBudget(limit: config.windowBytes)
        try await withPreparedMatrixModel(4) { url in
            let escaped = try await CoreMLBatchPipeline.withSession(compiledModelURL: url,
                inputColumns: plan.columnNames, outputName: "result", computeUnits: .cpuOnly,
                preprocessing: plan, lifetimeBudget: lifetime, workBudget: work,
                configuration: config) { session in
                for _ in 0..<2 {
                    let observations = PipelineObservations()
                    try await session.run(load: { $0.sequence < 3 ? source : nil }, consume: { seq, result in
                        #expect(await work.reservedBytes == work.limit)
                        try await observations.consumed(seq, result)
                    })
                    #expect(await observations.sequences == [0, 1, 2])
                    let rows = await observations.rows
                    #expect(rows == Array(repeating: source.originalRowIndices, count: 3))
                    #expect(await lifetime.reservedBytes == lifetime.limit)
                    #expect(await work.reservedBytes == 0)
                    try await work.withReservation(MemoryEstimate(capacities: [work.limit])) { }
                }
                await #expect(throws: PipelineTestFailure.consumer) {
                    try await session.run(load: { $0.sequence == 0 ? source : nil },
                        consume: { _, _ in throw PipelineTestFailure.consumer })
                }
                #expect(await work.reservedBytes == 0)
                try await session.run(load: { _ in nil }, consume: { _, _ in })
                return session
            }
            #expect(await lifetime.reservedBytes == 0)
            await #expect(throws: SwiftMLError.self) {
                try await escaped.run(load: { _ in source }, consume: { _, _ in })
            }
            #expect(await work.reservedBytes == 0)
        }
    }
}
