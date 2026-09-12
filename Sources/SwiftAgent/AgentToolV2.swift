import Foundation
import SwiftDataFrame

/// A property descriptor for an agent tool's structured JSON Schema parameters.
public struct AgentParameterProperty: Sendable, Codable, Equatable {
    /// The primitive JSON type (e.g. `"string"`, `"number"`, `"integer"`, `"boolean"`, `"array"`).
    public let type: String
    /// Human-readable explanation of the parameter's expected meaning and format.
    public let description: String
    /// Optional allowed enumeration values for string parameters.
    public let `enum`: [String]?
    /// Primitive type of elements if `type` is `"array"`.
    public let itemsType: String?

    /// Creates a new parameter property descriptor.
    ///
    /// - Parameters:
    ///   - type: JSON type string.
    ///   - description: Human-readable documentation for the model.
    ///   - enum: Optional array of valid literal values.
    ///   - itemsType: Element type when type is array.
    public init(
        type: String,
        description: String,
        enum: [String]? = nil,
        itemsType: String? = nil
    ) {
        self.type = type
        self.description = description
        self.enum = `enum`
        self.itemsType = itemsType
    }
}

/// Structured JSON Schema parameter specification for model function calling.
public struct AgentParameterSchema: Sendable, Codable, Equatable {
    /// The top-level schema type (typically `"object"`).
    public let type: String
    /// Key-value dictionary mapping parameter names to their properties.
    public let properties: [String: AgentParameterProperty]
    /// List of parameter names that must be provided in the tool call.
    public let required: [String]

    /// Creates a parameter schema for an agent tool.
    ///
    /// - Parameters:
    ///   - properties: Parameter definitions map.
    ///   - required: Names of mandatory parameters.
    public init(properties: [String: AgentParameterProperty], required: [String] = []) {
        self.type = "object"
        self.properties = properties
        self.required = required
    }

    /// An empty parameter schema requiring no arguments.
    public static var empty: AgentParameterSchema {
        AgentParameterSchema(properties: [:], required: [])
    }
}

/// Rich structured output produced by an autonomous agent tool invocation.
public struct AgentToolOutput: Sendable, Equatable {
    /// Human-readable textual representation or summary of the tool execution.
    public let text: String
    /// MIME type describing the format of the output (e.g. `"text/plain"`, `"application/json"`, `"image/svg+xml"`).
    public let mimeType: String
    /// Optional raw binary artifact payload produced by the tool (e.g. SVG image, Parquet bytes).
    public let data: Data?
    /// Execution wall-clock duration in milliseconds.
    public let durationMs: Double
    /// Additional contextual key-value metadata.
    public let metadata: [String: String]

    /// Initializes a new tool output payload.
    ///
    /// - Parameters:
    ///   - text: Textual result or summary.
    ///   - mimeType: Content MIME type (default: `"text/plain"`).
    ///   - data: Optional binary payload.
    ///   - durationMs: Wall-clock latency in milliseconds.
    ///   - metadata: Key-value context pairs.
    public init(
        text: String,
        mimeType: String = "text/plain",
        data: Data? = nil,
        durationMs: Double = 0.0,
        metadata: [String: String] = [:]
    ) {
        self.text = text
        self.mimeType = mimeType
        self.data = data
        self.durationMs = durationMs
        self.metadata = metadata
    }
}

/// An enhanced agent tool that accepts key-value argument dictionaries and yields rich `AgentToolOutput`.
public protocol StructuredAgentTool: AgentTool {
    /// The structured JSON Schema specification of parameters expected by this tool.
    var parameterSchema: AgentParameterSchema { get }

    /// Indicates whether executing this tool requires explicit human approval before running.
    var requiresApproval: Bool { get }

    /// Executes the tool using structured key-value arguments.
    ///
    /// - Parameter arguments: Parsed dictionary of arguments supplied by the model.
    /// - Returns: Rich structured execution output.
    /// - Throws: `AgentError` if validation or execution fails.
    func executeStructured(arguments: [String: String]) async throws -> AgentToolOutput
}

public extension StructuredAgentTool {
    /// Default fallback execution parsing a single text command.
    func execute(input: String) async throws -> String {
        let primaryKey = parameterSchema.required.first ?? parameterSchema.properties.keys.first ?? "input"
        var args: [String: String] = [primaryKey: input]
        args["input"] = input
        let out = try await executeStructured(arguments: args)
        return out.text
    }

    /// Default approval requirement is false.
    var requiresApproval: Bool { false }
}

public extension AgentTool {
    /// Default empty parameter schema for legacy tools.
    var parameterSchema: AgentParameterSchema {
        AgentParameterSchema(
            properties: [
                "input": AgentParameterProperty(type: "string", description: "Input parameter string for \(name).")
            ],
            required: ["input"]
        )
    }

    /// Default approval requirement for legacy tools.
    var requiresApproval: Bool { false }

    /// Executes tool with structured arguments by extracting the default input parameter.
    func executeStructured(arguments: [String: String]) async throws -> AgentToolOutput {
        let startTime = CFAbsoluteTimeGetCurrent()
        let inp = arguments["input"] ?? arguments["command"] ?? arguments.values.first ?? ""
        let res = try await execute(input: inp)
        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
        return AgentToolOutput(text: res, durationMs: elapsed)
    }

    /// Generates function declaration dictionary matching OpenAI/Anthropic/Apple function calling format.
    func toFunctionDeclaration() -> [String: String] {
        var jsonProps: [String] = []
        for (propName, prop) in parameterSchema.properties {
            var p = "\"\(propName)\": {\"type\": \"\(prop.type)\", \"description\": \"\(prop.description)\""
            if let enums = prop.enum {
                let enumList = enums.map { "\"\($0)\"" }.joined(separator: ", ")
                p += ", \"enum\": [\(enumList)]"
            }
            p += "}"
            jsonProps.append(p)
        }
        let reqList = parameterSchema.required.map { "\"\($0)\"" }.joined(separator: ", ")
        let schemaString = "{\"type\": \"object\", \"properties\": {\(jsonProps.joined(separator: ", "))}, \"required\": [\(reqList)]}"

        return [
            "name": name,
            "description": description,
            "parameters": schemaString
        ]
    }
}
