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

    @Test("Test Agent Evaluator filter and sample commands")
    func testAgentEvaluatorFilterAndSample() async throws {
        let ageCol = TypedColumn(name: "age", values: [20.0, 35.0, 50.0])
        let scoreCol = TypedColumn(name: "score", values: [80.0, 90.0, 95.0])
        let df = try DataFrame(columns: [ageCol, scoreCol])
        let eval = SwiftAgentEvaluator()

        // 1. Filter age > 30
        let filtered = try await eval.evaluate(command: "filter age > 30", on: df)
        #expect(filtered.rowCount == 2)

        // 2. Select age
        let selected = try await eval.evaluate(command: "select age", on: df)
        #expect(selected.columnNames == ["age"])

        // 3. Head 2
        let headDF = try await eval.evaluate(command: "head 2", on: df)
        #expect(headDF.rowCount == 2)

        // 4. Sample 2
        let sampled = try await eval.evaluate(command: "sample 2", on: df)
        #expect(sampled.rowCount == 2)

        // 5. Tail 1
        let tailDF = try await eval.evaluate(command: "tail 1", on: df)
        #expect(tailDF.rowCount == 1)

        // 6. Parenthesis / key-value syntaxes
        let filteredFunc = try await eval.evaluate(command: "filter(column: \"age\", condition: \"> 30\")", on: df)
        #expect(filteredFunc.rowCount == 2)

        let selectedFunc = try await eval.evaluate(command: "select(columns: [\"age\", \"score\"])", on: df)
        #expect(selectedFunc.columnNames == ["age", "score"])

        let sampledFunc = try await eval.evaluate(command: "sample(n: 2)", on: df)
        #expect(sampledFunc.rowCount == 2)

        let headFunc = try await eval.evaluate(command: "head(n: 1)", on: df)
        #expect(headFunc.rowCount == 1)

        let tailFunc = try await eval.evaluate(command: "tail(n: 1)", on: df)
        #expect(tailFunc.rowCount == 1)
    }

    @Test("Test Agent Evaluator rename command")
    func testAgentEvaluatorRename() async throws {
        let ageCol = TypedColumn(name: "age", values: [20.0, 35.0])
        let df = try DataFrame(columns: [ageCol])
        let eval = SwiftAgentEvaluator()

        let renamed = try await eval.evaluate(command: "rename age to years", on: df)
        #expect(renamed.columnNames == ["years"])
    }

    @Test("Test Agent Evaluator dropnulls command")
    func testAgentEvaluatorDropNulls() async throws {
        let ageCol = TypedColumn(name: "age", values: [20.0, nil, 50.0])
        let scoreCol = TypedColumn(name: "score", values: [80.0, 90.0, 95.0])
        let df = try DataFrame(columns: [ageCol, scoreCol])
        let eval = SwiftAgentEvaluator()

        let cleaned = try await eval.evaluate(command: "dropnulls", on: df)
        #expect(cleaned.rowCount == 2)
    }

    @Test("Test Agent Evaluator fillnulls command")
    func testAgentEvaluatorFillNulls() async throws {
        let scoreCol = TypedColumn(name: "score", values: [80.0, nil, 95.0])
        let df = try DataFrame(columns: [scoreCol])
        let eval = SwiftAgentEvaluator()

        let filled = try await eval.evaluate(command: "fillnulls score 0.0", on: df)
        guard let col = filled[column: "score", as: Double.self] else {
            Issue.record("Missing score column")
            return
        }
        #expect(col[1] == 0.0)
    }

    @Test("Test Agent Evaluator groupby command")
    func testAgentEvaluatorGroupBy() async throws {
        let categoryCol = TypedColumn(name: "category", values: ["A", "A", "B"])
        let valueCol = TypedColumn(name: "value", values: [10.0, 20.0, 30.0])
        let df = try DataFrame(columns: [categoryCol, valueCol])
        let eval = SwiftAgentEvaluator()

        let grouped = try await eval.evaluate(command: "groupby category mean", on: df)
        #expect(grouped.rowCount == 2)
    }

    @Test("Test Agent Evaluator unparseable command throws error")
    func testUnparseableCommandThrows() async {
        let col = TypedColumn(name: "A", values: [1.0, 2.0, 3.0])
        guard let df = try? DataFrame(columns: [col]) else { return }
        let eval = SwiftAgentEvaluator()

        await #expect(throws: AgentError.self) {
            _ = try await eval.evaluate(command: "invalid_gibberish_command", on: df)
        }

        await #expect(throws: AgentError.self) {
            _ = try await eval.evaluate(command: "groupby x bogus y", on: df)
        }
    }
}
