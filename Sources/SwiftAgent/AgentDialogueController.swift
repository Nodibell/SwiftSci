import Foundation
import SwiftDataFrame

#if canImport(Observation)
import Observation
#endif

/// Classification of message roles within an interactive agent dialogue session.
public enum AgentDialogueRole: String, Sendable, Codable, Equatable {
    /// Input prompt provided by the human user.
    case user
    /// Reasoning thought generated internally by the agent.
    case thought
    /// Invocation or observed output from an agent tool.
    case tool
    /// Final formulated answer presented to the user.
    case assistant
    /// System alert or error notice.
    case system
}

/// A structured message displayed in reactive SwiftUI chat interfaces like `AgentDialogueView`.
public struct AgentDialogueMessage: Sendable, Identifiable, Equatable {
    /// Unique message identifier.
    public let id: UUID
    /// Functional dialogue role.
    public let role: AgentDialogueRole
    /// Textual payload of the message.
    public var content: String
    /// Associated tool identifier if this message represents tool activity.
    public let toolName: String?
    /// Wall-clock latency of the tool invocation in milliseconds, if applicable.
    public let toolDurationMs: Double?
    /// Creation timestamp.
    public let timestamp: Date

    /// Initializes a dialogue message.
    ///
    /// - Parameters:
    ///   - id: Unique message UUID. Default is generated.
    ///   - role: Role classification.
    ///   - content: Text payload.
    ///   - toolName: Tool name if applicable.
    ///   - toolDurationMs: Execution latency if applicable.
    ///   - timestamp: Message timestamp.
    public init(
        id: UUID = UUID(),
        role: AgentDialogueRole,
        content: String,
        toolName: String? = nil,
        toolDurationMs: Double? = nil,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.toolName = toolName
        self.toolDurationMs = toolDurationMs
        self.timestamp = timestamp
    }
}

#if canImport(Observation)
/// A reactive `@Observable` controller providing 60 FPS state bindings for SwiftUI chat and dialogue views.
@Observable
@MainActor
public final class AgentDialogueController {
    /// Chronological list of dialogue messages in the active session.
    public var messages: [AgentDialogueMessage] = []

    /// Flag indicating whether the agent is currently reasoning or executing a tool.
    public var isThinking: Bool = false

    /// Currently running tool name, or `nil` if idle.
    public var activeToolName: String? = nil

    /// Pending human-in-the-loop approval ticket requiring user confirmation.
    public var pendingApproval: ApprovalRequest? = nil

    /// Current session error message, if any.
    public var errorMessage: String? = nil

    /// Initializes an empty dialogue controller.
    public init() {}

    /// Appends a new user message and processes it through the autonomous agent streaming pipeline.
    ///
    /// - Parameters:
    ///   - userQuery: The input prompt entered by the user.
    ///   - agent: The target `ReActAgent` instance.
    ///   - llm: Completion block generating language model tokens.
    public func send(
        userQuery: String,
        agent: ReActAgent,
        llm: @escaping @Sendable (String) async throws -> String
    ) async {
        let trimmed = userQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        errorMessage = nil
        messages.append(AgentDialogueMessage(role: .user, content: trimmed))
        isThinking = true
        activeToolName = nil

        let stream = await agent.stream(query: trimmed, llm: llm)

        for await event in stream {
            switch event {
            case .thoughtDelta(let delta):
                if let lastIdx = messages.indices.last, messages[lastIdx].role == .thought {
                    messages[lastIdx].content += delta
                } else {
                    messages.append(AgentDialogueMessage(role: .thought, content: delta))
                }

            case .toolCallScheduled(_, let tool, _):
                activeToolName = tool

            case .toolExecutionStarted(_, let tool):
                activeToolName = tool

            case .toolExecutionCompleted(_, let tool, let output, let durationMs):
                activeToolName = nil
                messages.append(
                    AgentDialogueMessage(
                        role: .tool,
                        content: output.text,
                        toolName: tool,
                        toolDurationMs: durationMs
                    )
                )

            case .answerDelta(let delta):
                activeToolName = nil
                if let lastIdx = messages.indices.last, messages[lastIdx].role == .assistant {
                    messages[lastIdx].content += delta
                } else {
                    messages.append(AgentDialogueMessage(role: .assistant, content: delta))
                }

            case .approvalRequested(let req):
                pendingApproval = req

            case .completed(let finalAnswer, _):
                isThinking = false
                activeToolName = nil
                if !messages.contains(where: { $0.role == .assistant && $0.content == finalAnswer }) {
                    messages.append(AgentDialogueMessage(role: .assistant, content: finalAnswer))
                }

            case .error(let err):
                isThinking = false
                activeToolName = nil
                errorMessage = err.localizedDescription
                messages.append(AgentDialogueMessage(role: .system, content: "Error: \(err.localizedDescription)"))
            }
        }

        isThinking = false
        activeToolName = nil
    }

    /// Clears all messages and resets dialogue state.
    public func clear() {
        messages.removeAll()
        isThinking = false
        activeToolName = nil
        pendingApproval = nil
        errorMessage = nil
    }
}
#endif
