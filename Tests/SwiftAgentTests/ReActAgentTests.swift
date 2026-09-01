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
        #expect(parsed2.action == nil)
        #expect(parsed2.finalAnswer == "The result is 42.")
    }

    @Test("CustomAgentTool and tool error handling in ReActAgent")
    func testCustomAgentToolAndErrors() async throws {
        let customTool = CustomAgentTool(name: "WeatherTool", description: "Provides weather") { input in
            return "Sunny in \(input)"
        }

        let agent = ReActAgent(tools: [customTool], maxSteps: 2)

        let mockLLM: @Sendable (String) async throws -> String = { prompt in
            if !prompt.contains("Previous History:") {
                return """
                Thought: Checking unknown tool first.
                Action: NonExistentTool
                Action Input: test
                """
            } else {
                return """
                Thought: Fallback to weather.
                Action: WeatherTool
                Action Input: Kyiv
                """
            }
        }

        let (answer, trace) = try await agent.run(query: "Weather", llm: mockLLM)
        #expect(trace.count == 2)
        #expect(trace[0].observation?.contains("Tool 'NonExistentTool' not recognized") == true)
        #expect(trace[1].observation == "Sunny in Kyiv")
        #expect(answer.contains("Sunny in Kyiv"))
    }

    @Test("ReActAgent findTool fuzzy matching")
    func testFindToolFuzzyMatching() async {
        let tool1 = CustomAgentTool(name: "DataFrameQuery", description: "Query dataframe") { _ in "" }
        let tool2 = CustomAgentTool(name: "Web_Search_Engine", description: "Search web") { _ in "" }
        let agent = ReActAgent(tools: [tool1, tool2])

        // Direct exact match
        #expect(await agent.findTool(named: "DataFrameQuery")?.name == "DataFrameQuery")

        // Case-insensitive / whitespace / underscores / hyphens match
        #expect(await agent.findTool(named: "dataframequery")?.name == "DataFrameQuery")
        #expect(await agent.findTool(named: "data_frame_query")?.name == "DataFrameQuery")
        #expect(await agent.findTool(named: "  dataframe-query  ")?.name == "DataFrameQuery")
        #expect(await agent.findTool(named: "web-search-engine")?.name == "Web_Search_Engine")
        #expect(await agent.findTool(named: "WEBSEARCHENGINE")?.name == "Web_Search_Engine")

        // Non existent
        #expect(await agent.findTool(named: "completely_unrelated_tool") == nil)
    }

    @Test("ReActAgent reaches maxSteps without final answer")
    func testMaxStepsReached() async throws {
        let tool = CustomAgentTool(name: "EchoTool", description: "Echo input") { "Echo: \($0)" }
        let agent = ReActAgent(tools: [tool], maxSteps: 2)

        let mockLLM: @Sendable (String) async throws -> String = { _ in
            return """
            Thought: Keep executing tool indefinitely.
            Action: EchoTool
            Action Input: loop
            """
        }

        let (answer, trace) = try await agent.run(query: "Loop forever", llm: mockLLM)
        #expect(trace.count == 2)
        #expect(answer == "Echo: loop")
    }
}
