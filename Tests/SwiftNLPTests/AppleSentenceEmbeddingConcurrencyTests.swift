import Foundation
import Testing
@testable import SwiftNLP

@Suite("AppleSentenceEmbeddingEngine concurrency")
struct AppleSentenceEmbeddingConcurrencyTests {
    @Test("Large batches and concurrent callers share one NLEmbedding safely and keep order")
    func largeConcurrentBatches() async throws {
        #if canImport(NaturalLanguage)
        guard let engine = try? AppleSentenceEmbeddingEngine() else { return }
        let texts = (0..<48).map { "Sample sentence number \($0) about topic \($0 % 5)." }
        let expected = try texts.map { try engine.embed($0) }

        let results = try await withThrowingTaskGroup(of: [[Double]].self) { group in
            for _ in 0..<6 { group.addTask { try await engine.embedBatch(texts) } }
            var all: [[[Double]]] = []
            for try await batch in group { all.append(batch) }
            return all
        }
        #expect(results.count == 6)
        for batch in results { #expect(batch == expected) }
        #expect(try await engine.embedBatch([]).isEmpty)
        #endif
    }

    @Test("Batch embedding honors cancellation")
    func cancellation() async throws {
        #if canImport(NaturalLanguage)
        guard let engine = try? AppleSentenceEmbeddingEngine() else { return }
        let task = Task { try await engine.embedBatch((0..<2000).map { "text \($0)" }) }
        task.cancel()
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        #endif
    }
}
