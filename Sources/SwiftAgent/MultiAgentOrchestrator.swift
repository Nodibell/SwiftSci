import Foundation
import SwiftDataFrame

/// A structured message passed through the agent communication bus.
public struct AgentMessage: Sendable, Identifiable, Equatable {
    /// Unique message identifier.
    public let id: UUID

    /// Identifying handle or name of the sending agent.
    public let sender: String

    /// The functional role of the agent (e.g. "planner", "analyst", "critic", "user").
    public let role: String

    /// Text payload or reasoning output of the message.
    public let content: String

    /// Message creation timestamp.
    public let timestamp: Date

    /// Optional parallel tool calls requested by the sender.
    public let toolCalls: [AgentToolCall]?

    /// Optional tool execution results attached to this message.
    public let toolResults: [AgentToolResult]?

    /// Initializes a new agent message.
    ///
    /// - Parameters:
    ///   - id: Unique message UUID. Default is generated.
    ///   - sender: Sender name or identifier.
    ///   - role: Role classification of sender.
    ///   - content: Main textual content of the message.
    ///   - timestamp: Message timestamp. Default is current date.
    ///   - toolCalls: Array of requested tool calls. Default is nil.
    ///   - toolResults: Array of tool output results. Default is nil.
    public init(
        id: UUID = UUID(),
        sender: String,
        role: String,
        content: String,
        timestamp: Date = Date(),
        toolCalls: [AgentToolCall]? = nil,
        toolResults: [AgentToolResult]? = nil
    ) {
        self.id = id
        self.sender = sender
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.toolCalls = toolCalls
        self.toolResults = toolResults
    }
}

/// A specification for an autonomous tool call requested by an agent.
public struct AgentToolCall: Sendable, Equatable {
    /// Unique identifier for this invocation instance.
    public let id: String

    /// Identifier of the tool to be invoked.
    public let toolName: String

    /// Input argument string passed to the tool.
    public let arguments: String

    /// Initializes a tool call specification.
    ///
    /// - Parameters:
    ///   - id: Call identifier. Default is UUID string.
    ///   - toolName: Name of the registered target tool.
    ///   - arguments: Tool input parameters.
    public init(id: String = UUID().uuidString, toolName: String, arguments: String) {
        self.id = id
        self.toolName = toolName
        self.arguments = arguments
    }
}

/// The observed output resulting from an executed agent tool call.
public struct AgentToolResult: Sendable, Equatable {
    /// Matching call identifier corresponding to `AgentToolCall.id`.
    public let callId: String

    /// Name of the tool that produced this result.
    public let toolName: String

    /// Output payload or error description string.
    public let output: String

    /// Indicates whether execution failed or encountered an error.
    public let isError: Bool

    /// Initializes a tool result.
    ///
    /// - Parameters:
    ///   - callId: The matching call identifier.
    ///   - toolName: The tool name.
    ///   - output: Output string from the tool.
    ///   - isError: Flag indicating if the tool failed. Default is false.
    public init(callId: String, toolName: String, output: String, isError: Bool = false) {
        self.callId = callId
        self.toolName = toolName
        self.output = output
        self.isError = isError
    }
}

/// An asynchronous event bus distributing messages between collaborating autonomous agents via `AsyncStream`.
///
/// ## Thread Safety
/// Implemented as an isolated Swift actor guaranteeing thread safety under Swift 6 strict concurrency.
///
/// ## Concurrency Management
/// Utilizes non-blocking Swift `AsyncStream` broadcasts to deliver events to all registered agent listeners.
public actor AgentMessageBus {
    private var continuations: [UUID: AsyncStream<AgentMessage>.Continuation] = [:]
    private var history: [AgentMessage] = []

    /// Initializes a new message bus instance.
    public init() {}

    /// Subscribes to the broadcast stream of agent messages.
    ///
    /// - Returns: A tuple consisting of the `AsyncStream<AgentMessage>` and its subscription UUID.
    public func subscribe() -> (stream: AsyncStream<AgentMessage>, subscriptionId: UUID) {
        let id = UUID()
        let (stream, continuation) = AsyncStream.makeStream(of: AgentMessage.self)
        continuations[id] = continuation
        continuation.onTermination = { [weak self] _ in
            Task { [weak self] in
                await self?.unsubscribe(id: id)
            }
        }
        return (stream, id)
    }

    /// Unsubscribes an active listener by its identifier.
    ///
    /// - Parameter id: The subscription UUID to remove.
    public func unsubscribe(id: UUID) {
        continuations.removeValue(forKey: id)
    }

    /// Publishes a message to all active subscriber streams and records it in history.
    ///
    /// - Parameter message: The `AgentMessage` to broadcast.
    public func publish(_ message: AgentMessage) {
        history.append(message)
        for (_, continuation) in continuations {
            continuation.yield(message)
        }
    }

    /// Returns the chronological history of all published messages.
    ///
    /// - Returns: An array of `AgentMessage` records.
    public func getHistory() -> [AgentMessage] {
        return history
    }

    /// Clears the recorded message history.
    public func clearHistory() {
        history.removeAll()
    }
}

/// A protocol defining a specialized autonomous agent capable of collaborative interaction.
public protocol SpecializedAgent: Sendable {
    /// The unique identifier of the agent.
    var name: String { get }

    /// The functional role of the agent (e.g. "Planner", "Analyst", "Critic").
    var role: String { get }

    /// Processes an incoming message in the context of prior conversation history.
    ///
    /// - Parameters:
    ///   - message: The latest message received.
    ///   - context: Preceding chronological messages on the bus.
    /// - Returns: An optional response message from this agent, or nil if no response is needed.
    /// - Throws: Any error encountered during reasoning or processing.
    func process(message: AgentMessage, context: [AgentMessage]) async throws -> AgentMessage?
}

/// Orchestrates collaborative execution across specialized autonomous agents over an asynchronous message bus.
///
/// `MultiAgentOrchestrator` coordinates multi-agent consensus loops, delegating sub-problems
/// to dedicated agents (e.g. Planner, Analyst, Critic) while broadcasting state transitions
/// and orchestrating parallel tool executions via `AgentMessageBus`.
///
/// ## Thread Safety
/// Implemented as an actor under strict Swift 6 concurrency, guaranteeing isolated state and thread-safe bus coordination.
///
/// ## Concurrency Management
/// Employs `AsyncStream` message loops and structured `withTaskGroup` parallel tool execution across CPU cores.
public actor MultiAgentOrchestrator {

    /// The asynchronous event bus coordinating all inter-agent messages.
    public let messageBus: AgentMessageBus

    /// Registered specialized agents keyed by agent name.
    public private(set) var agents: [String: any SpecializedAgent] = [:]

    /// Shared pool of tools available for parallel tool execution.
    public private(set) var tools: [String: any AgentTool] = [:]

    /// Maximum dialogue turns before concluding orchestrator execution.
    public let maxRounds: Int

    /// Initializes a new multi-agent orchestrator.
    ///
    /// - Parameters:
    ///   - messageBus: Custom message bus instance, or a new default instance.
    ///   - maxRounds: Maximum interaction turns per task. Default is 10.
    public init(messageBus: AgentMessageBus = AgentMessageBus(), maxRounds: Int = 10) {
        self.messageBus = messageBus
        self.maxRounds = maxRounds
    }

    /// Registers a specialized collaborating agent.
    ///
    /// - Parameter agent: The agent instance to add to the team.
    public func registerAgent(_ agent: any SpecializedAgent) {
        agents[agent.name] = agent
    }

    /// Registers a shared tool accessible by agents.
    ///
    /// - Parameter tool: The agent tool to register.
    public func registerTool(_ tool: any AgentTool) {
        tools[tool.name] = tool
    }

    /// Executes multiple tool calls concurrently via structured Swift Concurrency `withTaskGroup`.
    ///
    /// - Parameters:
    ///   - calls: An array of `AgentToolCall` specifications to execute.
    /// - Returns: An array of `AgentToolResult` entries matching each tool call.
    ///
    /// ## Concurrency Management
    /// Evaluates tool invocations concurrently across threads using `withTaskGroup`.
    ///
    /// ## Complexity
    /// \(O(\max_{i} T_i)\) wall-clock execution time instead of sequential \(\sum T_i\).
    public func executeToolsInParallel(_ calls: [AgentToolCall]) async -> [AgentToolResult] {
        let activeTools = self.tools
        return await withTaskGroup(of: AgentToolResult.self) { group in
            for call in calls {
                group.addTask {
                    guard let tool = activeTools[call.toolName] else {
                        return AgentToolResult(
                            callId: call.id,
                            toolName: call.toolName,
                            output: "Error: Tool '\(call.toolName)' is not registered.",
                            isError: true
                        )
                    }
                    do {
                        let result = try await tool.execute(input: call.arguments)
                        return AgentToolResult(
                            callId: call.id,
                            toolName: call.toolName,
                            output: result,
                            isError: false
                        )
                    } catch {
                        return AgentToolResult(
                            callId: call.id,
                            toolName: call.toolName,
                            output: "Execution failed: \(error.localizedDescription)",
                            isError: true
                        )
                    }
                }
            }

            var results: [AgentToolResult] = []
            for await res in group {
                results.append(res)
            }
            return results
        }
    }

    /// Solves a complex task through multi-agent collaboration and iterative consensus.
    ///
    /// - Parameters:
    ///   - taskPrompt: The high-level task or prompt for the multi-agent team.
    ///   - sequence: Optional explicit order of agent names to execute sequentially in each round.
    /// - Returns: A final synthesized response from the collaborative session.
    /// - Throws: Any error arising during message publishing or agent execution.
    ///
    /// ## Concurrency Management
    /// Structured async loop broadcasting turns via `AgentMessageBus` and resolving tool calls in parallel.
    public func run(
        taskPrompt: String,
        sequence: [String]? = nil
    ) async throws -> String {
        let initialMsg = AgentMessage(
            sender: "User",
            role: "user",
            content: taskPrompt
        )
        await messageBus.publish(initialMsg)

        let agentExecutionOrder: [any SpecializedAgent]
        if let seq = sequence {
            agentExecutionOrder = seq.compactMap { agents[$0] }
        } else {
            agentExecutionOrder = Array(agents.values)
        }

        var roundsCount = 0
        var lastContent = ""

        while roundsCount < maxRounds {
            roundsCount += 1
            var hadActivity = false

            for agent in agentExecutionOrder {
                let currentHistory = await messageBus.getHistory()
                guard let latest = currentHistory.last else { continue }

                if let response = try await agent.process(message: latest, context: currentHistory) {
                    hadActivity = true
                    lastContent = response.content

                    // If agent requested parallel tool calls, execute them concurrently
                    if let calls = response.toolCalls, !calls.isEmpty {
                        let results = await executeToolsInParallel(calls)
                        let toolResponseMsg = AgentMessage(
                            sender: agent.name,
                            role: agent.role,
                            content: response.content,
                            toolCalls: calls,
                            toolResults: results
                        )
                        await messageBus.publish(toolResponseMsg)
                    } else {
                        await messageBus.publish(response)
                    }

                    // Check for consensus / final answer signal
                    if response.content.localizedCaseInsensitiveContains("FINAL ANSWER:") {
                        let parts = response.content.components(separatedBy: "FINAL ANSWER:")
                        return parts.last?.trimmingCharacters(in: .whitespacesAndNewlines) ?? response.content
                    }
                }
            }

            if !hadActivity {
                break
            }
        }

        return lastContent.isEmpty ? "Task completed without explicit final answer." : lastContent
    }
}
