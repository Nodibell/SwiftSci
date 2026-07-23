import Testing
import Foundation
import SwiftDataFrame
@testable import SwiftAgent

@Suite("SwiftAgent Tests")
struct SwiftAgentTests {
    @Test("Test RAG summary generation")
    func testRAGSummary() throws {
        let col = TypedColumn(name: "A", values: [1.0, 2.0, 3.0])
        let df = try DataFrame(columns: [col])
        let gen = RAGContextGenerator()
        let summary = gen.generateSummary(df: df, name: "TestSet")

        #expect(summary.contains("TestSet Profile"))
        #expect(summary.contains("Rows: 3"))
    }

    @Test("Test Agent Evaluator")
    func testAgentEvaluator() async throws {
        let col = TypedColumn(name: "A", values: [1.0, 2.0, 3.0])
        let df = try DataFrame(columns: [col])
        let eval = SwiftAgentEvaluator()
        let result = try await eval.evaluate(command: "sample", on: df)

        #expect(result.rowCount <= 3)
    }
}
