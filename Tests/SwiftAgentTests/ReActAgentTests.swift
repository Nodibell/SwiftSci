import Testing
import Foundation
import SwiftDataFrame
import SwiftLLM
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

    @Test("ReActAgent runs with native LLMModel instance")
    func testReActAgentWithLocalLLMModel() async throws {
        struct MockModel: LLMModel {
            func generate(prompt: String, options: LLMOptions) async throws -> AsyncStream<String> {
                AsyncStream { continuation in
                    continuation.yield("""
                    Thought: Thinking locally on device.
                    Final Answer: Computed locally on Apple Silicon.
                    """)
                    continuation.finish()
                }
            }
        }

        let agent = ReActAgent()
        let model = MockModel()
        let (answer, trace) = try await agent.run(query: "Compute something", model: model)
        #expect(answer == "Computed locally on Apple Silicon.")
        #expect(trace.count == 1)
        #expect(trace[0].thought == "Thinking locally on device.")
    }

    @Test("ReActAgent tool execution times out when exceeding toolTimeoutSeconds")
    func testReActAgentToolExecutionTimeout() async throws {
        let slowTool = CustomAgentTool(name: "HangingTool", description: "Hangs indefinitely") { _ in
            try await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
            return "Should not finish"
        }

        let agent = ReActAgent(tools: [slowTool], maxSteps: 2, toolTimeoutSeconds: 0.1) // 100ms timeout

        let mockLLM: @Sendable (String) async throws -> String = { prompt in
            if !prompt.contains("Previous History:") {
                return """
                Thought: Testing hanging tool.
                Action: HangingTool
                Action Input: hang
                """
            } else {
                return """
                Thought: Tool timed out.
                Final Answer: Recovered from timeout safely.
                """
            }
        }

        let (answer, trace) = try await agent.run(query: "Test timeout", llm: mockLLM)
        #expect(trace.count == 2)
        #expect(trace[0].observation?.contains("timed out") == true)
        #expect(answer == "Recovered from timeout safely.")
    }

    @Test("ReActAgent with non-positive timeout disables timeout task")
    func testReActAgentZeroTimeout() async throws {
        let echoTool = CustomAgentTool(name: "Echo", description: "Echoes") { $0 }
        let agent = ReActAgent(tools: [echoTool], maxSteps: 2, toolTimeoutSeconds: 0.0)
        let mockLLM: @Sendable (String) async throws -> String = { prompt in
            if !prompt.contains("Previous History:") {
                return "Thought: test\nAction: Echo\nAction Input: hello"
            } else {
                return "Final Answer: done"
            }
        }
        let (answer, trace) = try await agent.run(query: "test", llm: mockLLM)
        #expect(answer == "done")
        #expect(trace[0].observation == "hello")
    }

    @Test("Gate 10: Agent resilience - malformed tool call does not break reasoning loop")
    func testMalformedToolCallDoesNotBreakLoop() async throws {
        struct CalcTool: StructuredAgentTool {
            let name = "Calculator"
            let description = "Performs arithmetic"
            let parameterSchema = AgentParameterSchema(
                properties: [
                    "expression": AgentParameterProperty(type: "string", description: "Math expression")
                ],
                required: ["expression"]
            )

            func executeStructured(arguments: [String: String]) async throws -> AgentToolOutput {
                let expr = arguments["expression"] ?? ""
                if expr == "2 + 2" {
                    return AgentToolOutput(text: "4")
                }
                return AgentToolOutput(text: "Evaluated \(expr)")
            }
        }

        let agent = ReActAgent(tools: [CalcTool()], maxSteps: 3)

        let mockLLM: @Sendable (String) async throws -> String = { prompt in
            if !prompt.contains("Previous History:") {
                // Step 1: Malformed JSON syntax
                return """
                Thought: I'll evaluate math with a malformed JSON payload.
                Action: Calculator
                Action Input: {"expression": "2 + 2", malformed_json
                """
            } else if !prompt.contains("Missing required parameter") {
                // Step 2: Valid JSON but missing required parameter
                return """
                Thought: Trying another tool call with missing required parameter.
                Action: Calculator
                Action Input: {"wrong_field": "123"}
                """
            } else {
                // Step 3: Successfully recovered and provides final answer
                return """
                Thought: Both previous calls failed gracefully without crashing.
                Final Answer: The reasoning loop continued uninterrupted.
                """
            }
        }

        let (answer, trace) = try await agent.run(query: "Calculate something", llm: mockLLM)

        #expect(answer == "The reasoning loop continued uninterrupted.")
        #expect(trace.count == 3)
        #expect(trace[0].observation?.contains("Malformed tool call") == true)
        #expect(trace[1].observation?.contains("Missing required parameter 'expression'") == true)
    }

    @Test("ReActAgent successfully executes structured tool call with valid JSON input")
    func testStructuredToolCallExecutionInReActAgent() async throws {
        struct MockStructuredTool: StructuredAgentTool {
            let name = "Multiplier"
            let description = "Multiplies two numbers"
            let parameterSchema = AgentParameterSchema(
                properties: [
                    "x": AgentParameterProperty(type: "number", description: "First number"),
                    "y": AgentParameterProperty(type: "number", description: "Second number")
                ],
                required: ["x", "y"]
            )

            func executeStructured(arguments: [String: String]) async throws -> AgentToolOutput {
                let x = Double(arguments["x"] ?? "0") ?? 0
                let y = Double(arguments["y"] ?? "0") ?? 0
                return AgentToolOutput(text: "\(Int(x * y))")
            }
        }

        let agent = ReActAgent(tools: [MockStructuredTool()], maxSteps: 2)
        let mockLLM: @Sendable (String) async throws -> String = { prompt in
            if !prompt.contains("Previous History:") {
                return """
                Thought: Multiplying numbers.
                Action: Multiplier
                Action Input: {"x": 6, "y": 7}
                """
            } else {
                return "Final Answer: The product is 42."
            }
        }

        let (ans, trace) = try await agent.run(query: "Multiply 6 and 7", llm: mockLLM)
        #expect(ans == "The product is 42.")
        #expect(trace[0].observation == "42")
    }

    @Test("AgentParameterSchema and SchemaValidationError comprehensive coverage")
    func testAgentParameterSchemaComprehensive() {
        // Test empty schema
        let emptySchema = AgentParameterSchema.empty
        #expect(emptySchema.properties.isEmpty)
        #expect(emptySchema.required.isEmpty)

        // Test property with enum and itemsType
        let prop = AgentParameterProperty(type: "array", description: "Tags", enum: ["tag1", "tag2"], itemsType: "string")
        #expect(prop.enum == ["tag1", "tag2"])
        #expect(prop.itemsType == "string")

        // Schema with enum validation
        let schema = AgentParameterSchema(
            properties: [
                "mode": AgentParameterProperty(type: "string", description: "Operating mode", enum: ["fast", "accurate"]),
                "count": AgentParameterProperty(type: "number", description: "Count"),
                "flag": AgentParameterProperty(type: "boolean", description: "Flag")
            ],
            required: ["mode"]
        )

        // Test valid JSON with NSNumber, Bool, and string
        let validJSON = "{\"mode\": \"fast\", \"count\": 10, \"flag\": true, \"other\": [1, 2]}"
        let res = schema.parseAndValidate(jsonString: validJSON)
        switch res {
        case .success(let dict):
            #expect(dict["mode"] == "fast")
            #expect(dict["count"] == "10")
            #expect(dict["flag"] == "true")
        case .failure:
            Issue.record("Expected successful parse")
        }

        // Test invalid enum value
        let invalidEnumJSON = "{\"mode\": \"unsupported\"}"
        let resEnum = schema.parseAndValidate(jsonString: invalidEnumJSON)
        switch resEnum {
        case .failure(let err):
            #expect(err == .invalidEnumValue(param: "mode", value: "unsupported", allowed: ["fast", "accurate"]))
            #expect(err.errorDescription?.contains("Invalid value 'unsupported'") == true)
        case .success:
            Issue.record("Expected enum validation failure")
        }

        // Test all SchemaValidationError descriptions
        let errMissing = SchemaValidationError.missingRequired("apiKey")
        #expect(errMissing.errorDescription == "Missing required parameter 'apiKey'")

        let errJSON = SchemaValidationError.invalidJSON("bad_json")
        #expect(errJSON.errorDescription == "Malformed JSON syntax in tool arguments: 'bad_json'")

        let errUTF8 = SchemaValidationError.utf8ConversionFailed
        #expect(errUTF8.errorDescription == "Malformed input: cannot convert to UTF-8")
    }

    @Test("G-038: parseResponse handles Gemma 4 <|channel>thought blocks (inline, multi-block, empty, unclosed)")
    func testParseGemma4ChannelThoughtBlocks() async {
        let agent = ReActAgent()

        // Inline directive immediately following <channel|>
        let inline = "<|channel>thought\nLet me plan.<channel|>Action: EchoTool\nAction Input: hello"
        let parsedInline = await agent.parseResponse(inline)
        #expect(parsedInline.thought == "Let me plan.")
        #expect(parsedInline.action == "EchoTool")
        #expect(parsedInline.actionInput == "hello")
        #expect(parsedInline.finalAnswer == nil)

        // Multiple blocks + inline Final Answer
        let multi = "<|channel>thought\nStep one.<channel|>\n<|channel>thought\nStep two.<channel|>Final Answer: 42"
        let parsedMulti = await agent.parseResponse(multi)
        #expect(parsedMulti.thought == "Step one.\nStep two.")
        #expect(parsedMulti.action == nil)
        #expect(parsedMulti.finalAnswer == "42")

        // Empty thought block
        let empty = "<|channel>thought\n<channel|>Final Answer: Clean"
        let parsedEmpty = await agent.parseResponse(empty)
        #expect(parsedEmpty.thought.isEmpty)
        #expect(parsedEmpty.finalAnswer == "Clean")

        // Unclosed thought block with half-generated Action: must not execute or become finalAnswer
        let unclosed = "<|channel>thought\nIncomplete reasoning\nAction: DangerTool\nAction Input: boom"
        let parsedUnclosed = await agent.parseResponse(unclosed)
        #expect(parsedUnclosed.thought == "Incomplete reasoning\nAction: DangerTool\nAction Input: boom")
        #expect(parsedUnclosed.action == nil)
        #expect(parsedUnclosed.actionInput == nil)
        #expect(parsedUnclosed.finalAnswer == nil)
    }

    @Test("G-038: parseResponse handles <think>...</think> closed and unclosed blocks")
    func testParseXMLThinkBlocks() async {
        let agent = ReActAgent()

        let closedInline = "<think>Compute sum</think>Final Answer: 30"
        let parsedClosed = await agent.parseResponse(closedInline)
        #expect(parsedClosed.thought == "Compute sum")
        #expect(parsedClosed.finalAnswer == "30")

        let unclosed = "<think>Still thinking\nAction: EchoTool\nAction Input: partial"
        let parsedUnclosed = await agent.parseResponse(unclosed)
        #expect(parsedUnclosed.thought == "Still thinking\nAction: EchoTool\nAction Input: partial")
        #expect(parsedUnclosed.action == nil)
        #expect(parsedUnclosed.actionInput == nil)
        #expect(parsedUnclosed.finalAnswer == nil)
    }

    @Test("G-052: parseResponse stops collecting Action Input at Observation: (case-insensitive)")
    func testParseActionInputStopsAtObservation() async {
        let agent = ReActAgent()

        let singleLine = """
        Thought: Querying data.
        Action: DataFrameQuery
        Action Input: filter: age > 25
        observation: Result (3 rows)
        extra hallucinated observation line
        """
        let parsedSingle = await agent.parseResponse(singleLine)
        #expect(parsedSingle.action == "DataFrameQuery")
        #expect(parsedSingle.actionInput == "filter: age > 25")
        #expect(parsedSingle.finalAnswer == nil)

        let multiLine = """
        Action: Calc
        Action Input: line 1
        line 2
        Observation: 99
        more ignored text
        """
        let parsedMulti = await agent.parseResponse(multiLine)
        #expect(parsedMulti.action == "Calc")
        #expect(parsedMulti.actionInput == "line 1\nline 2")
    }

    @Test("G-052: ReActAgent executes Action when Action precedes hallucinated Final Answer in same turn")
    func testActionPriorityOverHallucinatedFinalAnswer() async throws {
        let tool = CustomAgentTool(name: "EchoTool", description: "Echo input") { "Real: \($0)" }
        let agent = ReActAgent(tools: [tool], maxSteps: 3)

        let mockLLM: @Sendable (String) async throws -> String = { prompt in
            if !prompt.contains("Previous History:") {
                return """
                Thought: Run EchoTool first.
                Action: EchoTool
                Action Input: ping
                Observation: Hallucinated observation
                Thought: Hallucinated follow-up
                Final Answer: Hallucinated answer
                """
            } else {
                return """
                Thought: Observed real tool output.
                Final Answer: Verified answer
                """
            }
        }

        let (answer, trace) = try await agent.run(query: "Test priority", llm: mockLLM)
        #expect(trace.count == 2)
        #expect(trace[0].action == "EchoTool")
        #expect(trace[0].actionInput == "ping")
        #expect(trace[0].observation == "Real: ping")
        #expect(answer == "Verified answer")
    }
}
