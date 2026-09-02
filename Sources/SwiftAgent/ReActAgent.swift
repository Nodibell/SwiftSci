import SwiftLLM
import Foundation
import SwiftDataFrame

/// A tool callable by an autonomous ReAct agent.
public protocol AgentTool: Sendable {
    /// The unique identifier name of the tool.
    var name: String { get }
    /// A human-readable description explaining what the tool does and expected input format.
    var description: String { get }
    /// Executes the tool logic with the given string input and returns the observation string.
    func execute(input: String) async throws -> String
}

/// A custom tool wrapping a Swift async closure.
public struct CustomAgentTool: AgentTool, Sendable {
    /// The unique identifier name of the tool.
    public let name: String
    /// A human-readable description explaining what the tool does.
    public let description: String
    private let handler: @Sendable (String) async throws -> String

    /// Creates a new closure-based agent tool.
    ///
    /// - Parameters:
    ///   - name: Unique tool name.
    ///   - description: Description of tool functionality.
    ///   - handler: Async execution block.
    public init(name: String, description: String, handler: @escaping @Sendable (String) async throws -> String) {
        self.name = name
        self.description = description
        self.handler = handler
    }

    /// Executes the custom tool handler.
    public func execute(input: String) async throws -> String {
        try await handler(input)
    }
}

/// An agent tool for executing sandboxed DataFrame query and transformation expressions.
public struct DataFrameAgentTool: AgentTool, Sendable {
    /// The unique identifier name of the tool.
    public let name: String = "DataFrameQuery"
    /// A human-readable description explaining what the tool does and expected input format.
    public let description: String = "Executes transformation commands on the active DataFrame (e.g., 'filter: age > 25', 'groupBy: department, mean, salary', 'head: 5')."
    
    private let evaluator: SwiftAgentEvaluator
    private let dataframe: DataFrame

    /// Initializes a DataFrame query agent tool.
    ///
    /// - Parameters:
    ///   - dataframe: Target DataFrame instance.
    ///   - evaluator: Sandbox evaluator (default: new instance).
    public init(dataframe: DataFrame, evaluator: SwiftAgentEvaluator = SwiftAgentEvaluator()) {
        self.dataframe = dataframe
        self.evaluator = evaluator
    }

    /// Evaluates the DataFrame command and returns a string summary of the resulting DataFrame.
    public func execute(input: String) async throws -> String {
        let result = try await evaluator.evaluate(command: input, on: dataframe)
        var out = "Result (\(result.rowCount) rows):\n"
        for col in result.columnNames {
            guard let anyCol = result[column: col] else { continue }
            let sampleValues = (0..<Swift.min(3, result.rowCount)).map { r in
                if let val = anyCol.value(at: r) {
                    return "\(val)"
                } else {
                    return "nil"
                }
            }
            out += "- \(col): [\(sampleValues.joined(separator: ", "))\(result.rowCount > 3 ? ", ..." : "")]\n"
        }
        return out
    }
}

/// A single step in an autonomous ReAct reasoning trajectory.
public struct AgentStep: Sendable, Equatable {
    /// The agent's reasoning step.
    public let thought: String
    /// The selected tool name to invoke, if any.
    public let action: String?
    /// The input string provided to the tool.
    public let actionInput: String?
    /// The result observed from tool execution.
    public let observation: String?

    /// Creates an AgentStep.
    public init(thought: String, action: String? = nil, actionInput: String? = nil, observation: String? = nil) {
        self.thought = thought
        self.action = action
        self.actionInput = actionInput
        self.observation = observation
    }
}

/// Autonomous ReAct (Reasoning + Acting) Agent executing multi-step decision loops over tool pipelines.
///
/// `ReActAgent` orchestrates dynamic Tool usage by prompting an LLM or reasoning engine to iteratively
/// generate `Thought -> Action -> Action Input -> Observation` sequences until reaching a `Final Answer`.
public actor ReActAgent {
    /// Mapping of registered tool names to tool instances.
    public private(set) var tools: [String: any AgentTool]
    /// Maximum allowed reasoning steps before termination.
    public let maxSteps: Int

    /// Initializes a ReAct Agent with available tools.
    ///
    /// - Parameters:
    ///   - tools: List of initial tools available to the agent.
    ///   - maxSteps: Maximum step limit to prevent infinite loops (default: 10).
    public init(tools: [any AgentTool] = [], maxSteps: Int = 10) {
        var toolMap: [String: any AgentTool] = [:]
        for tool in tools {
            toolMap[tool.name] = tool
        }
        self.tools = toolMap
        self.maxSteps = maxSteps
    }

    /// Registers a new tool with the agent.
    ///
    /// - Parameter tool: Tool instance to add.
    public func registerTool(_ tool: any AgentTool) {
        tools[tool.name] = tool
    }

    /// Parses a raw LLM text response into Thought, Action, Action Input, or Final Answer components.
    ///
    /// - Parameter response: Raw output from LLM.
    /// - Returns: Extracted step tuple.
    public func parseResponse(_ response: String) -> (thought: String, action: String?, actionInput: String?, finalAnswer: String?) {
        var thought = ""
        var action: String? = nil
        var actionInput: String? = nil
        var finalAnswer: String? = nil

        let lines = response.components(separatedBy: .newlines)
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.lowercased().hasPrefix("thought:") {
                let idx = trimmed.index(trimmed.startIndex, offsetBy: 8)
                thought = String(trimmed[idx...]).trimmingCharacters(in: .whitespaces)
            } else if trimmed.lowercased().hasPrefix("final answer:") {
                let idx = trimmed.index(trimmed.startIndex, offsetBy: 13)
                finalAnswer = String(trimmed[idx...]).trimmingCharacters(in: .whitespaces)
            } else if trimmed.lowercased().hasPrefix("action:") {
                let idx = trimmed.index(trimmed.startIndex, offsetBy: 7)
                action = String(trimmed[idx...]).trimmingCharacters(in: .whitespaces)
            } else if trimmed.lowercased().hasPrefix("action input:") {
                let idx = trimmed.index(trimmed.startIndex, offsetBy: 13)
                actionInput = String(trimmed[idx...]).trimmingCharacters(in: .whitespaces)
            }
        }

        if finalAnswer == nil && action == nil && !response.isEmpty {
            // Fallback: entire response is final answer
            finalAnswer = response.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        return (thought: thought, action: action, actionInput: actionInput, finalAnswer: finalAnswer)
    }

    /// Executes the autonomous ReAct reasoning loop.
    ///
    /// - Parameters:
    ///   - query: The target user question or goal.
    ///   - llm: Async block producing model completions given a prompt.
    /// - Returns: Final answer and complete trajectory trace of agent steps.
    public func run(
        query: String,
        llm: @Sendable (String) async throws -> String
    ) async throws -> (finalAnswer: String, trace: [AgentStep]) {
        var trace: [AgentStep] = []

        for _ in 0..<maxSteps {
            var prompt = "You are an autonomous AI analyst answering the following question: '\(query)'.\n\n"
            prompt += "Available Tools:\n"
            for (_, tool) in tools {
                prompt += "- \(tool.name): \(tool.description)\n"
            }
            prompt += "\nFormat instructions:\nThought: [reasoning]\nAction: [tool name]\nAction Input: [input]\nOr:\nThought: [reasoning]\nFinal Answer: [result]\n\n"

            if !trace.isEmpty {
                prompt += "Previous History:\n"
                for step in trace {
                    prompt += "Thought: \(step.thought)\n"
                    if let act = step.action, let inp = step.actionInput {
                        prompt += "Action: \(act)\nAction Input: \(inp)\n"
                    }
                    if let obs = step.observation {
                        prompt += "Observation: \(obs)\n"
                    }
                }
            }

            let response = try await llm(prompt)
            let parsed = parseResponse(response)

            if let answer = parsed.finalAnswer {
                let step = AgentStep(thought: parsed.thought, observation: answer)
                trace.append(step)
                return (finalAnswer: answer, trace: trace)
            }

            guard let actName = parsed.action, let actTool = findTool(named: actName) else {
                let fallbackStep = AgentStep(thought: parsed.thought, observation: "Error: Tool '\(parsed.action ?? "nil")' not recognized.")
                trace.append(fallbackStep)
                continue
            }

            let input = parsed.actionInput ?? ""
            let obs = try await actTool.execute(input: input)
            let step = AgentStep(thought: parsed.thought, action: actTool.name, actionInput: input, observation: obs)
            trace.append(step)
        }

        return (finalAnswer: trace.last?.observation ?? "Max steps reached without conclusive answer.", trace: trace)
    }

    /// Resolves tool by exact or fuzzy name matching (case/punctuation-insensitive).
    public func findTool(named name: String) -> (any AgentTool)? {
        if let direct = tools[name] { return direct }
        let clean = name.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: " ", with: "")
        for (tName, tool) in tools {
            let tClean = tName.lowercased()
                .replacingOccurrences(of: "_", with: "")
                .replacingOccurrences(of: "-", with: "")
                .replacingOccurrences(of: " ", with: "")
            if tClean == clean || tClean.contains(clean) || clean.contains(tClean) {
                return tool
            }
        }
        return nil
    }

    /// Executes the ReAct reasoning cycle directly using a native `LLMModel` (e.g. `TransformerDecoder`).
    ///
    /// - Parameters:
    ///   - query: Analytical question or directive to execute.
    ///   - model: A local LLMModel instance executing natively on Apple Silicon.
    ///   - options: Inference options (temperature, topP, maxTokens).
    /// - Returns: Tuple with final answer and complete reasoning step trace.
    public func run(
        query: String,
        model: any LLMModel,
        options: LLMOptions = LLMOptions(temperature: 0.2, maxTokens: 256)
    ) async throws -> (finalAnswer: String, trace: [AgentStep]) {
        return try await run(query: query) { prompt in
            let stream = try await model.generate(prompt: prompt, options: options)
            var generated = ""
            for await token in stream {
                generated += token
            }
            return generated
        }
    }

}
