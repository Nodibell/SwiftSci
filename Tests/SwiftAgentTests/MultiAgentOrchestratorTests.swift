import Testing
import Foundation
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
}
