import Foundation

/// Represents an interactive approval request requiring human intervention before executing a sensitive tool.
public struct ApprovalRequest: Sendable, Identifiable, Equatable {
    /// Unique identifier for this approval ticket.
    public let id: String
    /// The name of the tool requesting execution.
    public let toolName: String
    /// The arguments passed to the tool.
    public let arguments: [String: String]
    /// Human-readable explanation of why approval is needed or what changes will occur.
    public let explanation: String
    /// Timestamp when the request was initiated.
    public let createdAt: Date

    /// Creates an approval request.
    ///
    /// - Parameters:
    ///   - id: Unique ticket ID. Default is a new UUID string.
    ///   - toolName: Tool name.
    ///   - arguments: Tool input arguments.
    ///   - explanation: Explanatory summary.
    ///   - createdAt: Request initiation timestamp.
    public init(
        id: String = UUID().uuidString,
        toolName: String,
        arguments: [String: String],
        explanation: String,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.toolName = toolName
        self.arguments = arguments
        self.explanation = explanation
        self.createdAt = createdAt
    }
}

/// Real-time streaming event emitted during an autonomous agent reasoning and execution trajectory.
public enum AgentStreamEvent: Sendable, Equatable {
    /// Incremental reasoning thought token emitted by the model.
    case thoughtDelta(String)

    /// The agent scheduled an invocation of a registered tool.
    case toolCallScheduled(callId: String, tool: String, arguments: [String: String])

    /// Execution of the tool has begun.
    case toolExecutionStarted(callId: String, tool: String)

    /// Execution of the tool finished successfully with output and latency measurement.
    case toolExecutionCompleted(callId: String, tool: String, output: AgentToolOutput, durationMs: Double)

    /// Incremental final response token emitted to the user.
    case answerDelta(String)

    /// Execution paused waiting for human approval.
    case approvalRequested(ApprovalRequest)

    /// Trajectory terminated with a final answer and complete audit trail.
    case completed(finalAnswer: String, trajectory: [AgentStep])

    /// An unrecoverable error occurred during reasoning or tool execution.
    case error(AgentError)
}
