import XCTest
@testable import SwiftAgent
@testable import SwiftDataFrame
@testable import SwiftStats

final class SwiftAgentOmniTests: XCTestCase {

    // MARK: - 1. Structured Tool & Schema Tests

    func testStructuredAgentToolSchema() async throws {
        let statsTool = SwiftSciToolbox.StatisticsInspectionTool()
        XCTAssertEqual(statsTool.name, "StatisticsInspection")
        let schema = statsTool.parameterSchema
        XCTAssertEqual(schema.type, "object")
        XCTAssertTrue(schema.required.contains("action"))
        XCTAssertTrue(schema.required.contains("columnA"))

        let decl = statsTool.toFunctionDeclaration()
        XCTAssertEqual(decl["name"], "StatisticsInspection")
        XCTAssertTrue(decl["parameters"]?.contains("columnA") == true)
        XCTAssertTrue(decl["parameters"]?.contains("enum") == true)
    }

    func testAgentParameterPropertyAndSchemaCodableAndDefaults() throws {
        let prop = AgentParameterProperty(
            type: "array",
            description: "List of items",
            enum: ["opt1", "opt2"],
            itemsType: "string"
        )
        XCTAssertEqual(prop.type, "array")
        XCTAssertEqual(prop.description, "List of items")
        XCTAssertEqual(prop.enum, ["opt1", "opt2"])
        XCTAssertEqual(prop.itemsType, "string")

        // Codable round-trip
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let data = try encoder.encode(prop)
        let decodedProp = try decoder.decode(AgentParameterProperty.self, from: data)
        XCTAssertEqual(prop, decodedProp)

        // Schema empty
        let emptySchema = AgentParameterSchema.empty
        XCTAssertEqual(emptySchema.type, "object")
        XCTAssertEqual(emptySchema.properties.count, 0)
        XCTAssertEqual(emptySchema.required.count, 0)

        let schemaData = try encoder.encode(emptySchema)
        let decodedSchema = try decoder.decode(AgentParameterSchema.self, from: schemaData)
        XCTAssertEqual(emptySchema, decodedSchema)

        // Tool output
        let rawData = "payload".data(using: .utf8)
        let out = AgentToolOutput(
            text: "test",
            mimeType: "text/csv",
            data: rawData,
            durationMs: 42.5,
            metadata: ["key": "val"]
        )
        XCTAssertEqual(out.text, "test")
        XCTAssertEqual(out.mimeType, "text/csv")
        XCTAssertEqual(out.data, rawData)
        XCTAssertEqual(out.durationMs, 42.5)
        XCTAssertEqual(out.metadata["key"], "val")
    }

    func testLegacyAgentToolDefaultExtensions() async throws {
        let legacy = CustomAgentTool(name: "LegacyEcho", description: "Echoes input") { inp in
            return "Echo: \(inp)"
        }

        XCTAssertFalse(legacy.requiresApproval)
        let schema = legacy.parameterSchema
        XCTAssertTrue(schema.properties.keys.contains("input"))
        XCTAssertEqual(schema.required, ["input"])

        let outInput = try await legacy.executeStructured(arguments: ["input": "hello"])
        XCTAssertEqual(outInput.text, "Echo: hello")

        let outCommand = try await legacy.executeStructured(arguments: ["command": "run"])
        XCTAssertEqual(outCommand.text, "Echo: run")

        let outFirstVal = try await legacy.executeStructured(arguments: ["custom": "foobar"])
        XCTAssertEqual(outFirstVal.text, "Echo: foobar")

        let outEmpty = try await legacy.executeStructured(arguments: [:])
        XCTAssertEqual(outEmpty.text, "Echo: ")

        let decl = legacy.toFunctionDeclaration()
        XCTAssertEqual(decl["name"], "LegacyEcho")
        XCTAssertTrue(decl["parameters"]?.contains("input") == true)
    }

    func testStructuredAgentToolDefaultExecuteFallback() async throws {
        let tool = SwiftSciToolbox.NLPSentimentTool()
        let res = try await tool.execute(input: "Superb results!")
        XCTAssertTrue(res.contains("VADER Sentiment Analysis"))
        XCTAssertFalse(tool.requiresApproval)
    }

    // MARK: - 2. SwiftSciToolbox Omni-Module Tests

    func testSwiftSciToolboxStatistics() async throws {
        let tool = SwiftSciToolbox.StatisticsInspectionTool()

        // Test descriptive summary
        let resSummary = try await tool.executeStructured(arguments: [
            "action": "summary",
            "columnA": "1.0, 2.0, 3.0, 4.0, 5.0"
        ])
        XCTAssertTrue(resSummary.text.contains("Mean: 3.0000"))
        XCTAssertTrue(resSummary.durationMs >= 0)

        // Test Welch's t-test
        let resTTest = try await tool.executeStructured(arguments: [
            "action": "ttest",
            "columnA": "10.0, 11.0, 12.0, 10.5, 11.5",
            "columnB": "20.0, 21.0, 22.0, 20.5, 21.5"
        ])
        XCTAssertTrue(resTTest.text.contains("Welch's Two-Sample t-Test"))
        XCTAssertTrue(resTTest.text.contains("Significant (α=0.05): true"))

        // Test correlation
        let resCorr = try await tool.executeStructured(arguments: [
            "action": "correlation",
            "columnA": "1.0, 2.0, 3.0, 4.0, 5.0",
            "columnB": "2.0, 4.0, 6.0, 8.0, 10.0"
        ])
        XCTAssertTrue(resCorr.text.contains("Pearson r: 1.0000"))
    }

    func testSwiftSciToolboxStatisticsWithDataFrameAndErrors() async throws {
        let colSales = TypedColumn<Double>(name: "sales", values: [10.0, 20.0, 30.0, 40.0])
        let colProfit = TypedColumn<Double>(name: "profit", values: [2.0, 4.0, 6.0, 8.0])
        let df = try DataFrame(columns: [colSales, colProfit])
        let tool = SwiftSciToolbox.StatisticsInspectionTool(dataframeProvider: { df })

        // Extract values from DataFrame columns
        let summaryRes = try await tool.executeStructured(arguments: [
            "action": "summary",
            "columnA": "sales"
        ])
        XCTAssertTrue(summaryRes.text.contains("Mean: 25.0000"))

        let corrRes = try await tool.executeStructured(arguments: [
            "action": "correlation",
            "columnA": "sales",
            "columnB": "profit"
        ])
        XCTAssertTrue(corrRes.text.contains("Pearson r: 1.0000"))

        let ttestRes = try await tool.executeStructured(arguments: [
            "action": "ttest",
            "columnA": "sales",
            "columnB": "profit"
        ])
        XCTAssertTrue(ttestRes.text.contains("Welch's Two-Sample t-Test"))

        // Error: missing primary column values
        do {
            _ = try await tool.executeStructured(arguments: ["columnA": "nonexistent"])
            XCTFail("Expected error on missing column")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("No numeric values found"))
        }

        // Error: ttest without columnB
        do {
            _ = try await tool.executeStructured(arguments: ["action": "ttest", "columnA": "sales", "columnB": ""])
            XCTFail("Expected error on missing columnB for ttest")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Welch's t-test requires valid secondary column"))
        }

        // Error: correlation without columnB
        do {
            _ = try await tool.executeStructured(arguments: ["action": "correlation", "columnA": "sales", "columnB": ""])
            XCTFail("Expected error on missing columnB for correlation")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Correlation requires valid secondary column"))
        }

        // Fallback execute(input:)
        do {
            _ = try await tool.execute(input: "1.0, 2.0, 3.0")
            XCTFail("Expected error on missing columnA")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("No numeric values found"))
        }
    }

    func testSwiftSciToolboxDataDrift() async throws {
        let tool = SwiftSciToolbox.DataDriftInspectionTool()
        let res = try await tool.executeStructured(arguments: [
            "reference": "1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0",
            "production": "1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0"
        ])
        XCTAssertTrue(res.text.contains("Population Stability Index (PSI)"))
        XCTAssertTrue(res.text.contains("Wasserstein Distance (W1)"))
        XCTAssertTrue(res.text.contains("STABLE DISTRIBUTION"))

        let resDrift = try await tool.executeStructured(arguments: [
            "reference": "1.0, 2.0, 3.0, 4.0, 5.0",
            "production": "50.0, 60.0, 70.0, 80.0, 90.0"
        ])
        XCTAssertTrue(resDrift.text.contains("CRITICAL DRIFT DETECTED"))

        // Error branches
        do {
            _ = try await tool.executeStructured(arguments: ["reference": "1,2"])
            XCTFail("Expected error on missing production")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Both 'reference' and 'production' parameters are required"))
        }

        do {
            _ = try await tool.executeStructured(arguments: ["reference": "abc", "production": "def"])
            XCTFail("Expected error on non-numeric distributions")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Could not parse numeric distributions"))
        }

        // Fallback execute(input:)
        do {
            _ = try await tool.execute(input: "reference=1,2,3")
            XCTFail("Expected error on missing production")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Both 'reference' and 'production' parameters are required"))
        }
    }

    func testSwiftSciToolboxForecasting() async throws {
        let tool = SwiftSciToolbox.TimeSeriesForecastingTool()
        let series = [10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 20.0, 22.0, 21.0, 23.0, 25.0, 27.0]
        let seriesStr = series.map { "\($0)" }.joined(separator: ", ")

        // Specified horizon
        let res = try await tool.executeStructured(arguments: [
            "values": seriesStr,
            "horizon": "3"
        ])
        XCTAssertTrue(res.text.contains("ARIMA(1,1,1) Forecast"))

        // Default horizon
        let resDefault = try await tool.executeStructured(arguments: [
            "values": seriesStr
        ])
        XCTAssertTrue(resDefault.text.contains("ARIMA(1,1,1) Forecast (h=12)"))

        // Error: missing values
        do {
            _ = try await tool.executeStructured(arguments: [:])
            XCTFail("Expected error on missing values")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Parameter 'values' is required"))
        }

        // Error: <10 observations
        do {
            _ = try await tool.executeStructured(arguments: ["values": "1.0, 2.0, 3.0"])
            XCTFail("Expected error on <10 observations")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("requires at least 10 historical observations"))
        }

        // Fallback execute
        let fb = try await tool.execute(input: seriesStr)
        XCTAssertTrue(fb.contains("ARIMA(1,1,1) Forecast"))
    }

    func testSwiftSciToolboxNLPSentiment() async throws {
        let tool = SwiftSciToolbox.NLPSentimentTool()
        let resPos = try await tool.executeStructured(arguments: [
            "text": "SwiftSci 3.8.0 provides extraordinary performance, elegance, and reliability!"
        ])
        XCTAssertTrue(resPos.text.contains("Positive 🟢"))

        let resNeg = try await tool.executeStructured(arguments: [
            "text": "This is a terrible disaster and completely unacceptable crash."
        ])
        XCTAssertTrue(resNeg.text.contains("Negative 🔴"))

        let resNeu = try await tool.executeStructured(arguments: [
            "text": "The file is located on the filesystem."
        ])
        XCTAssertTrue(resNeu.text.contains("Neutral ⚪"))

        // Error: missing text
        do {
            _ = try await tool.executeStructured(arguments: [:])
            XCTFail("Expected error on missing text")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Parameter 'text' is required"))
        }

        // Fallback execute
        let fb = try await tool.execute(input: "Great work!")
        XCTAssertTrue(fb.contains("Positive 🟢"))
    }

    func testSwiftSciToolboxDatabaseQuery() async throws {
        let tool = SwiftSciToolbox.SafeDatabaseQueryTool(databasePath: ":memory:")
        XCTAssertFalse(tool.requiresApproval)

        // Mutating statement blocked by sentry
        let blockedWords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]
        for w in blockedWords {
            do {
                _ = try await tool.executeStructured(arguments: [
                    "query": "\(w) TABLE users;"
                ])
                XCTFail("Expected security sentry error on \(w) statement")
            } catch {
                XCTAssertTrue(error.localizedDescription.contains("Security Violation"))
            }
        }

        // Missing query parameter
        do {
            _ = try await tool.executeStructured(arguments: [:])
            XCTFail("Expected error on missing query parameter")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Parameter 'query' is required"))
        }

        // Valid read-only query returning > 3 rows
        let selectMulti = try await tool.executeStructured(arguments: [
            "query": "SELECT 1 as num UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5;"
        ])
        XCTAssertTrue(selectMulti.text.contains("5 rows returned"))
        XCTAssertTrue(selectMulti.text.contains("... (2 more rows)"))

        // Fallback execute
        let fb = try await tool.execute(input: "SELECT 1 as num;")
        XCTAssertTrue(fb.contains("Query executed successfully"))
    }

    func testSwiftSciToolboxStandardTools() {
        let tools = SwiftSciToolbox.standardTools()
        XCTAssertEqual(tools.count, 5)
        XCTAssertTrue(tools.contains(where: { $0.name == "StatisticsInspection" }))
        XCTAssertTrue(tools.contains(where: { $0.name == "DataDriftInspection" }))
        XCTAssertTrue(tools.contains(where: { $0.name == "TimeSeriesForecasting" }))
        XCTAssertTrue(tools.contains(where: { $0.name == "NLPSentiment" }))
        XCTAssertTrue(tools.contains(where: { $0.name == "DatabaseQuery" }))
    }

    // MARK: - 3. Memory & Vector Memory Tests

    func testSlidingWindowMemory() async throws {
        let memory = SlidingWindowMemory(maxRecords: 3)
        let maxRecs = await memory.maxRecords
        XCTAssertEqual(maxRecs, 3)

        try await memory.record(content: "First turn", role: "user", metadata: ["k": "v"])
        try await memory.record(content: "Second turn", role: "assistant")
        try await memory.record(content: "Third turn", role: "user")
        try await memory.record(content: "Fourth turn", role: "assistant")

        let context = try await memory.retrieveContext(query: "anything", limit: 10)
        XCTAssertEqual(context.count, 3)
        XCTAssertEqual(context[0].content, "Second turn")
        XCTAssertEqual(context[1].content, "Third turn")
        XCTAssertEqual(context[2].content, "Fourth turn")

        try await memory.clear()
        let emptyContext = try await memory.retrieveContext(query: "anything", limit: 10)
        XCTAssertEqual(emptyContext.count, 0)
    }

    func testSemanticVectorMemory() async throws {
        let embedder: @Sendable (String) async throws -> [Double] = { text in
            let bytes = Array(text.utf8)
            let b0 = Double(bytes.first ?? 0)
            let b1 = Double(bytes.last ?? 0)
            let b2 = Double(bytes.count)
            let b3 = Double(text.contains("apple") ? 100 : 0)
            return [b0, b1, b2, b3]
        }

        let memory = SemanticVectorMemory(embedder: embedder)
        // Retrieve on empty
        let emptyBefore = try await memory.retrieveContext(query: "anything")
        XCTAssertTrue(emptyBefore.isEmpty)

        try await memory.record(content: "We use apple silicon GPU kernels", role: "assistant")
        try await memory.record(content: "The weather in Kyiv is cloudy", role: "user")

        let retrieved = try await memory.retrieveContext(query: "apple metal kernels", limit: 1)
        XCTAssertEqual(retrieved.count, 1)
        XCTAssertEqual(retrieved.first?.content, "We use apple silicon GPU kernels")

        try await memory.clear()
        let emptyAfter = try await memory.retrieveContext(query: "apple metal kernels", limit: 1)
        XCTAssertTrue(emptyAfter.isEmpty)
    }

    // MARK: - 4. Streaming, Loop Sentry & Approval Tests

    private actor CallCounter {
        var count = 0
        func next() -> Int {
            count += 1
            return count
        }
    }

    private struct ApprovalRequiringTool: StructuredAgentTool, Sendable {
        let name = "DangerousReboot"
        let description = "Reboots the system"
        let parameterSchema = AgentParameterSchema.empty
        let requiresApproval = true
        func executeStructured(arguments: [String: String]) async throws -> AgentToolOutput {
            AgentToolOutput(text: "Reboot triggered")
        }
    }

    private struct CrashingTool: StructuredAgentTool, Sendable {
        let name = "CrashTool"
        let description = "Throws an error"
        let parameterSchema = AgentParameterSchema.empty
        let requiresApproval = false
        func executeStructured(arguments: [String: String]) async throws -> AgentToolOutput {
            throw AgentError.executionFailed("Hardware sensor offline")
        }
    }

    func testReActAgentStreamingAndLoopSentry() async throws {
        let tool = CustomAgentTool(name: "PingTool", description: "Replies with pong") { _ in
            return "pong"
        }
        let agent = ReActAgent(tools: [tool], maxSteps: 5)

        let counter = CallCounter()
        let llm: @Sendable (String) async throws -> String = { prompt in
            let callCount = await counter.next()
            if callCount <= 3 {
                return "Thought: Need to ping\nAction: PingTool\nAction Input: test"
            } else {
                return "Thought: Loop detected, answering.\nFinal Answer: Ping completed."
            }
        }

        let stream = await agent.stream(query: "Run ping", llm: llm)
        var eventsReceived: [AgentStreamEvent] = []

        for await event in stream {
            eventsReceived.append(event)
        }

        XCTAssertTrue(eventsReceived.contains(where: {
            if case .completed(let ans, _) = $0 {
                return ans.contains("Ping completed")
            }
            return false
        }))
    }

    func testReActAgentStreamingWithMemoryApprovalAndErrors() async throws {
        let memory = SlidingWindowMemory(maxRecords: 5)
        try await memory.record(content: "Previous fact: System is online", role: "user")

        let approvalTool = ApprovalRequiringTool()
        let crashTool = CrashingTool()
        let agent = ReActAgent(tools: [approvalTool, crashTool], maxSteps: 4, memory: memory)

        let counter = CallCounter()
        let llm: @Sendable (String) async throws -> String = { prompt in
            let call = await counter.next()
            if call == 1 {
                // Verify memory context was injected into prompt
                XCTAssertTrue(prompt.contains("Relevant Long-Term Memory:"))
                return "Thought: Must reboot\nAction: DangerousReboot\nAction Input: now"
            } else if call == 2 {
                return "Thought: Now test crash\nAction: CrashTool\nAction Input: none"
            } else if call == 3 {
                return "Thought: Call unknown\nAction: NonExistentTool\nAction Input: xyz"
            } else {
                return "Final Answer: Trajectory finished safely."
            }
        }

        let stream = await agent.stream(query: "Execute pipeline", llm: llm)
        var events: [AgentStreamEvent] = []
        for await ev in stream {
            events.append(ev)
        }

        // Verify approval request was emitted
        XCTAssertTrue(events.contains(where: {
            if case .approvalRequested(let req) = $0 {
                return req.toolName == "DangerousReboot"
            }
            return false
        }))

        // Verify completed with final answer
        XCTAssertTrue(events.contains(where: {
            if case .completed(let ans, let trace) = $0 {
                return ans.contains("Trajectory finished safely") && trace.count >= 3
            }
            return false
        }))
    }

    func testReActAgentStreamingMaxStepsFallback() async throws {
        let agent = ReActAgent(tools: [], maxSteps: 2)
        let llm: @Sendable (String) async throws -> String = { _ in
            return "Thought: Need to act\nAction: NonExistentTool\nAction Input: test"
        }

        let (finalAnswer, trace) = try await agent.run(query: "Endless task", llm: llm)
        XCTAssertTrue(finalAnswer.contains("not recognized"))
        XCTAssertEqual(trace.count, 2)
    }

    func testReActAgentRunRethrowsErrors() async throws {
        let agent = ReActAgent(tools: [], maxSteps: 2)
        let llm: @Sendable (String) async throws -> String = { _ in
            throw AgentError.executionFailed("Model inference failed")
        }

        do {
            _ = try await agent.run(query: "Test fail", llm: llm)
            XCTFail("Expected error to be rethrown")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Model inference failed"))
        }
    }

    // MARK: - 5. AgentStreamEvent & ApprovalRequest Tests

    func testApprovalRequestAndStreamEventEquatables() {
        let req1 = ApprovalRequest(id: "1", toolName: "T", arguments: ["a": "b"], explanation: "Exp")
        let req2 = ApprovalRequest(id: "1", toolName: "T", arguments: ["a": "b"], explanation: "Exp", createdAt: req1.createdAt)
        XCTAssertEqual(req1, req2)
        XCTAssertEqual(req1.id, "1")
        XCTAssertEqual(req1.toolName, "T")
        XCTAssertEqual(req1.arguments, ["a": "b"])
        XCTAssertEqual(req1.explanation, "Exp")

        // Equatable cases of AgentStreamEvent
        XCTAssertEqual(AgentStreamEvent.thoughtDelta("t"), AgentStreamEvent.thoughtDelta("t"))
        XCTAssertNotEqual(AgentStreamEvent.thoughtDelta("t"), AgentStreamEvent.thoughtDelta("t2"))

        XCTAssertEqual(
            AgentStreamEvent.toolCallScheduled(callId: "1", tool: "T", arguments: [:]),
            AgentStreamEvent.toolCallScheduled(callId: "1", tool: "T", arguments: [:])
        )
        XCTAssertEqual(
            AgentStreamEvent.toolExecutionStarted(callId: "1", tool: "T"),
            AgentStreamEvent.toolExecutionStarted(callId: "1", tool: "T")
        )
        let out = AgentToolOutput(text: "out")
        XCTAssertEqual(
            AgentStreamEvent.toolExecutionCompleted(callId: "1", tool: "T", output: out, durationMs: 1.0),
            AgentStreamEvent.toolExecutionCompleted(callId: "1", tool: "T", output: out, durationMs: 1.0)
        )
        XCTAssertEqual(AgentStreamEvent.answerDelta("ans"), AgentStreamEvent.answerDelta("ans"))
        XCTAssertEqual(AgentStreamEvent.approvalRequested(req1), AgentStreamEvent.approvalRequested(req2))
        XCTAssertEqual(
            AgentStreamEvent.completed(finalAnswer: "done", trajectory: []),
            AgentStreamEvent.completed(finalAnswer: "done", trajectory: [])
        )
        XCTAssertEqual(
            AgentStreamEvent.error(.toolTimeout(tool: "T", seconds: 5.0)),
            AgentStreamEvent.error(.toolTimeout(tool: "T", seconds: 5.0))
        )
    }

    // MARK: - 6. Dialogue Controller Tests

    #if canImport(Observation)
    @MainActor
    func testAgentDialogueController() async throws {
        let controller = AgentDialogueController()
        XCTAssertFalse(controller.isThinking)
        XCTAssertEqual(controller.messages.count, 0)

        // Empty user query does nothing
        await controller.send(userQuery: "   ", agent: ReActAgent(tools: []), llm: { _ in "" })
        XCTAssertEqual(controller.messages.count, 0)

        // Full interaction with thoughts, tool calls, and deltas
        let tool = CustomAgentTool(name: "CalcTool", description: "Calculator") { _ in "42" }
        let agent = ReActAgent(tools: [tool], maxSteps: 3)

        let counter = CallCounter()
        let llm: @Sendable (String) async throws -> String = { _ in
            let c = await counter.next()
            if c == 1 {
                return "Thought: Calculating\nAction: CalcTool\nAction Input: 6*7"
            } else {
                return "Thought: Got the answer\nFinal Answer: The answer is 42."
            }
        }

        await controller.send(userQuery: "What is 6*7?", agent: agent, llm: llm)

        XCTAssertFalse(controller.isThinking)
        XCTAssertNil(controller.activeToolName)
        XCTAssertTrue(controller.messages.count >= 4) // user, thought, tool, thought, assistant

        XCTAssertEqual(controller.messages.first?.role, .user)
        XCTAssertEqual(controller.messages.first?.content, "What is 6*7?")
        XCTAssertTrue(controller.messages.contains(where: { $0.role == .tool && $0.content == "42" && $0.toolName == "CalcTool" }))
        XCTAssertTrue(controller.messages.contains(where: { $0.role == .assistant && $0.content.contains("The answer is 42") }))

        controller.clear()
        XCTAssertEqual(controller.messages.count, 0)
        XCTAssertNil(controller.pendingApproval)
        XCTAssertNil(controller.errorMessage)
    }

    @MainActor
    func testAgentDialogueControllerErrorAndApproval() async throws {
        let controller = AgentDialogueController()
        let approvalTool = ApprovalRequiringTool()
        let agent = ReActAgent(tools: [approvalTool], maxSteps: 2)

        let llm: @Sendable (String) async throws -> String = { _ in
            return "Thought: Requesting reboot\nAction: DangerousReboot\nAction Input: now"
        }

        await controller.send(userQuery: "Reboot system", agent: agent, llm: llm)
        XCTAssertNotNil(controller.pendingApproval)
        XCTAssertEqual(controller.pendingApproval?.toolName, "DangerousReboot")

        // Test error handling in controller
        let failingAgent = ReActAgent(tools: [], maxSteps: 2)
        await controller.send(userQuery: "Fail query", agent: failingAgent, llm: { _ in
            throw AgentError.executionFailed("Network drop")
        })

        XCTAssertNotNil(controller.errorMessage)
        XCTAssertTrue(controller.messages.contains(where: { $0.role == .system && $0.content.contains("Network drop") }))

        // Dialogue Roles & Message Model
        let roles: [AgentDialogueRole] = [.user, .thought, .tool, .assistant, .system]
        for r in roles {
            XCTAssertFalse(r.rawValue.isEmpty)
        }

        let customMsg = AgentDialogueMessage(
            role: .tool,
            content: "payload",
            toolName: "CustomTool",
            toolDurationMs: 12.3
        )
        XCTAssertEqual(customMsg.role, .tool)
        XCTAssertEqual(customMsg.content, "payload")
        XCTAssertEqual(customMsg.toolName, "CustomTool")
        XCTAssertEqual(customMsg.toolDurationMs, 12.3)
    }
    #endif
}
