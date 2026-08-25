import Testing
import Foundation
import SwiftDataFrame
@testable import SwiftAgent

@Suite("ReAct Agent Reasoning Loop Tests")
struct ReActAgentTests {

    @Test("ReActAgent executes multi-step reasoning trajectory with tools")
    func testReActAgentExecution() async throws {
        let df = try DataFrame(columns: [
            TypedColumn(name: "name", values: ["Alice", "Bob", "Charlie", "David"]),
            TypedColumn(name: "age", values: [25.0, 30.0, 35.0, 40.0]),
            TypedColumn(name: "salary", values: [50000.0, 60000.0, 70000.0, 80000.0])
        ])

        let dfTool = DataFrameAgentTool(dataframe: df)
        let agent = ReActAgent(tools: [dfTool], maxSteps: 5)

        let mockLLM: @Sendable (String) async throws -> String = { prompt in
            if !prompt.contains("Previous History:") {
                return """
                Thought: I should check employees with age greater than 28.
                Action: DataFrameQuery
                Action Input: filter: age > 28
                """
            } else {
                return """
                Thought: I now have the filtered rows.
                Final Answer: Found 3 employees older than 28 (Bob, Charlie, David).
                """
            }
        }

        let (answer, trace) = try await agent.run(query: "Find employees older than 28", llm: mockLLM)

        #expect(answer.contains("Found 3 employees"))
        #expect(trace.count == 2)
        #expect(trace[0].action == "DataFrameQuery")
        #expect(trace[0].observation != nil)
    }

    @Test("ReActAgent response parsing handles various formats")
    func testResponseParsing() async throws {
        let agent = ReActAgent()

        let response1 = """
        Thought: Need to calculate sum.
        Action: Calculator
        Action Input: 10 + 20
        """
        let parsed1 = await agent.parseResponse(response1)
        #expect(parsed1.thought == "Need to calculate sum.")
        #expect(parsed1.action == "Calculator")
        #expect(parsed1.actionInput == "10 + 20")
        #expect(parsed1.finalAnswer == nil)

        let response2 = """
        Thought: Done thinking.
        Final Answer: The result is 42.
        """
        let parsed2 = await agent.parseResponse(response2)
        #expect(parsed2.thought == "Done thinking.")
        #expect(parsed2.finalAnswer == "The result is 42.")
    }
}
