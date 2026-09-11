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
    }

    func testSwiftSciToolboxForecasting() async throws {
        let tool = SwiftSciToolbox.TimeSeriesForecastingTool()
        let series = [10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 20.0, 22.0, 21.0, 23.0, 25.0, 27.0]
        let seriesStr = series.map { "\($0)" }.joined(separator: ", ")

        let res = try await tool.executeStructured(arguments: [
            "values": seriesStr,
            "horizon": "3"
        ])
        XCTAssertTrue(res.text.contains("ARIMA(1,1,1) Forecast"))
    }

    func testSwiftSciToolboxNLPSentiment() async throws {
        let tool = SwiftSciToolbox.NLPSentimentTool()
        let res = try await tool.executeStructured(arguments: [
            "text": "SwiftSci 3.8.0 provides extraordinary performance, elegance, and reliability!"
        ])
        XCTAssertTrue(res.text.contains("VADER Sentiment Analysis"))
        XCTAssertTrue(res.text.contains("Positive 🟢"))
    }

    func testSwiftSciToolboxDatabaseQuery() async throws {
        let tool = SwiftSciToolbox.SafeDatabaseQueryTool(databasePath: ":memory:")

        // Mutating statement blocked by sentry
        do {
            _ = try await tool.executeStructured(arguments: [
                "query": "DROP TABLE users;"
            ])
            XCTFail("Expected security sentry error on DROP statement")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Security Violation"))
        }

        // Safe SELECT query
        let res = try await tool.executeStructured(arguments: [
            "query": "SELECT 1 AS num, 'SwiftSci' AS framework;"
        ])
        XCTAssertTrue(res.text.contains("Query executed successfully"))
        XCTAssertTrue(res.text.contains("Columns: [num, framework]"))
    }

    // MARK: - 3. Agent Memory Tests

    func testSlidingWindowMemory() async throws {
        let memory = SlidingWindowMemory(maxRecords: 3)
        try await memory.record(content: "First turn", role: "user")
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
        // Simple hash-based 4D embedder for testing
        let embedder: @Sendable (String) async throws -> [Double] = { text in
            let bytes = Array(text.utf8)
            let b0 = Double(bytes.first ?? 0)
            let b1 = Double(bytes.last ?? 0)
            let b2 = Double(bytes.count)
            let b3 = Double(text.contains("apple") ? 100 : 0)
            return [b0, b1, b2, b3]
        }

        let memory = SemanticVectorMemory(embedder: embedder)
        try await memory.record(content: "We use apple silicon GPU kernels", role: "assistant")
        try await memory.record(content: "The weather in Kyiv is cloudy", role: "user")

        let retrieved = try await memory.retrieveContext(query: "apple metal kernels", limit: 1)
        XCTAssertEqual(retrieved.count, 1)
        XCTAssertEqual(retrieved.first?.content, "We use apple silicon GPU kernels")
    }

    // MARK: - 4. Streaming & Loop Sentry Tests

    private actor CallCounter {
        var count = 0
        func next() -> Int {
            count += 1
            return count
        }
    }

    func testReActAgentStreamingAndLoopSentry() async throws {
        let tool = CustomAgentTool(name: "PingTool", description: "Replies with pong") { _ in
            return "pong"
        }
        let agent = ReActAgent(tools: [tool], maxSteps: 5)

        // Mock LLM that loops twice calling PingTool, receives sentry warning on 3rd, and provides Final Answer
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

    // MARK: - 5. Dialogue Controller Tests

    #if canImport(Observation)
    @MainActor
    func testAgentDialogueController() async throws {
        let controller = AgentDialogueController()
        XCTAssertFalse(controller.isThinking)
        XCTAssertEqual(controller.messages.count, 0)

        let agent = ReActAgent(tools: [], maxSteps: 2)
        let llm: @Sendable (String) async throws -> String = { _ in
            return "Thought: Direct response\nFinal Answer: Hello from SwiftAgent 3.8.0!"
        }

        await controller.send(userQuery: "Hello!", agent: agent, llm: llm)

        XCTAssertFalse(controller.isThinking)
        XCTAssertEqual(controller.messages.count, 3) // user, thought, assistant
        XCTAssertEqual(controller.messages[0].role, .user)
        XCTAssertEqual(controller.messages[0].content, "Hello!")
        XCTAssertEqual(controller.messages[2].role, .assistant)
        XCTAssertEqual(controller.messages[2].content, "Hello from SwiftAgent 3.8.0!")

        controller.clear()
        XCTAssertEqual(controller.messages.count, 0)
    }
    #endif
}
