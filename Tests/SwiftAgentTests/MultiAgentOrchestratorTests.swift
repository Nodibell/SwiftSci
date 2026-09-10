import Testing
import Foundation
import SwiftDataFrame
@testable import SwiftAgent

@Suite("MultiAgent Orchestrator & Message Bus Tests")
struct MultiAgentOrchestratorTests {

    @Test("AgentMessageBus publishes messages to async subscriber streams")
    func testMessageBusPublishSubscribe() async throws {
        let bus = AgentMessageBus()
        let (stream, subId) = await bus.subscribe()

        let msg1 = AgentMessage(sender: "AgentA", role: "analyst", content: "Observation 1")
        await bus.publish(msg1)

        var received: [AgentMessage] = []
        for await msg in stream {
            received.append(msg)
            if received.count == 1 {
                break
            }
        }

        #expect(received.count == 1)
        #expect(received[0].sender == "AgentA")
        #expect(received[0].content == "Observation 1")

        let history = await bus.getHistory()
        #expect(history.count == 1)

        await bus.unsubscribe(id: subId)
    }

    @Test("MultiAgentOrchestrator executes multiple tools in parallel")
    func testParallelToolExecution() async throws {
        let orchestrator = MultiAgentOrchestrator()

        let toolA = CustomAgentTool(name: "DoubleNum", description: "Doubles integer input") { input in
            guard let val = Int(input) else { return "0" }
            return "\(val * 2)"
        }
        let toolB = CustomAgentTool(name: "SquareNum", description: "Squares integer input") { input in
            guard let val = Int(input) else { return "0" }
            return "\(val * val)"
        }

        await orchestrator.registerTool(toolA)
        await orchestrator.registerTool(toolB)

        let calls = [
            AgentToolCall(id: "call_1", toolName: "DoubleNum", arguments: "21"),
            AgentToolCall(id: "call_2", toolName: "SquareNum", arguments: "9"),
            AgentToolCall(id: "call_3", toolName: "MissingTool", arguments: "123")
        ]

        let results = await orchestrator.executeToolsInParallel(calls)
        #expect(results.count == 3)

        let resMap = Dictionary(uniqueKeysWithValues: results.map { ($0.callId, $0) })
        #expect(resMap["call_1"]?.output == "42")
        #expect(resMap["call_1"]?.isError == false)

        #expect(resMap["call_2"]?.output == "81")
        #expect(resMap["call_2"]?.isError == false)

        #expect(resMap["call_3"]?.isError == true)
    }

    struct MockPlanner: SpecializedAgent {
        let name = "Planner"
        let role = "planner"

        func process(message: AgentMessage, context: [AgentMessage]) async throws -> AgentMessage? {
            if message.role == "user" {
                return AgentMessage(
                    sender: name,
                    role: role,
                    content: "Plan: calculate 10 + 20 and then verify.",
                    toolCalls: [
                        AgentToolCall(id: "c1", toolName: "Calculator", arguments: "10+20")
                    ]
                )
            }
            return nil
        }
    }

    struct MockCritic: SpecializedAgent {
        let name = "Critic"
        let role = "critic"

        func process(message: AgentMessage, context: [AgentMessage]) async throws -> AgentMessage? {
            if let results = message.toolResults, let first = results.first {
                return AgentMessage(
                    sender: name,
                    role: role,
                    content: "Verification passed with result \(first.output). FINAL ANSWER: \(first.output)"
                )
            }
            return nil
        }
    }

    @Test("Collaborative MultiAgent consensus loop with parallel tool calls")
    func testCollaborativeLoop() async throws {
        let orchestrator = MultiAgentOrchestrator(maxRounds: 5)

        let calcTool = CustomAgentTool(name: "Calculator", description: "Performs math") { input in
            if input == "10+20" { return "30" }
            return "0"
        }
        await orchestrator.registerTool(calcTool)
        await orchestrator.registerAgent(MockPlanner())
        await orchestrator.registerAgent(MockCritic())

        let answer = try await orchestrator.run(
            taskPrompt: "Compute 10 + 20",
            sequence: ["Planner", "Critic"]
        )

        #expect(answer.contains("30"))
    }

    @Test("executeToolsInParallel handles throwing tool gracefully")
    func testParallelToolExecutionErrorThrown() async throws {
        let orchestrator = MultiAgentOrchestrator()
        let throwingTool = CustomAgentTool(name: "ThrowingTool", description: "Throws") { _ in
            throw AgentError.executionFailed("Simulated crash")
        }
        await orchestrator.registerTool(throwingTool)
        let results = await orchestrator.executeToolsInParallel([
            AgentToolCall(id: "c1", toolName: "ThrowingTool", arguments: "abc")
        ])
        #expect(results.count == 1)
        #expect(results[0].isError == true)
        #expect(results[0].output.contains("Execution failed"))
    }

    struct SimpleSpeaker: SpecializedAgent {
        let name = "Speaker"
        let role = "speaker"
        func process(message: AgentMessage, context: [AgentMessage]) async throws -> AgentMessage? {
            if message.role == "user" {
                return AgentMessage(sender: name, role: role, content: "Hello! FINAL ANSWER: 42")
            }
            return nil
        }
    }

    @Test("MultiAgentOrchestrator with default sequence and direct response without tools")
    func testDefaultSequence() async throws {
        let orchestrator = MultiAgentOrchestrator(maxRounds: 3)
        await orchestrator.registerAgent(SimpleSpeaker())
        let answer = try await orchestrator.run(taskPrompt: "Hello")
        #expect(answer == "42")
    }

    struct SilentAgent: SpecializedAgent {
        let name = "Silent"
        let role = "silent"
        func process(message: AgentMessage, context: [AgentMessage]) async throws -> AgentMessage? {
            nil
        }
    }

    @Test("MultiAgentOrchestrator breaks early when no agent produces response")
    func testInactiveExit() async throws {
        let orchestrator = MultiAgentOrchestrator(maxRounds: 5)
        await orchestrator.registerAgent(SilentAgent())
        let answer = try await orchestrator.run(taskPrompt: "Nothing")
        #expect(answer == "Task completed without explicit final answer.")
    }

    @Test("SwiftAgentEvaluator lineage recording, clearing and AgentError.toolTimeout")
    func testEvaluatorLineageAndTimeout() async throws {
        let evaluator = SwiftAgentEvaluator()
        let df = try DataFrame(columns: [TypedColumn(name: "val", values: [1.0, 2.0, 3.0])])
        let res = try await evaluator.evaluate(command: "filter: val > 1", on: df)
        #expect(res.rowCount == 2)

        let lineage = await evaluator.lineage
        #expect(lineage.count == 1)
        #expect(lineage[0].inputRows == 3)
        #expect(lineage[0].outputRows == 2)
        #expect(lineage[0].operation.contains("filter"))

        await evaluator.clearLineage()
        let emptyLineage = await evaluator.lineage
        #expect(emptyLineage.isEmpty)

        let timeoutErr = AgentError.toolTimeout(tool: "calculator", seconds: 15.0)
        #expect(timeoutErr.errorDescription?.contains("calculator") == true)
        #expect(timeoutErr.errorDescription?.contains("15.0") == true)
    }
}
