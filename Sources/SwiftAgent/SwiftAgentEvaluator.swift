import Foundation
import SwiftDataFrame

/// Errors thrown by SwiftAgent evaluation sandbox.
public enum AgentError: Error, LocalizedError, Equatable {
    case unparseable(String)
    case executionFailed(String)

    /// The error description.
    public var errorDescription: String? {
        switch self {
        case .unparseable(let cmd):
            return "Failed to parse agent command: '\(cmd)'."
        case .executionFailed(let msg):
            return "Agent execution failed: \(msg)"
        }
    }
}

/// Structured Agent AST representation for sandboxed execution.
public enum AgentCommand: Sendable {
    case filter(column: String, condition: FilterCondition)
    case sample(n: Int)
    case select(columns: [String])
    case head(n: Int)
    case tail(n: Int)
}

/// RAG Context summary generator for dataframes.
public struct RAGContextGenerator: Sendable {
    /// Creates a new instance.
    public init() {}

    /// Generates token-efficient Markdown summary of DataFrame for AI Analyst system prompt.
    public func generateSummary(df: DataFrame, name: String = "Dataset") -> String {
        var summary = "## \(name) Profile\n"
        summary += "- Rows: \(df.rowCount), Columns: \(df.columnNames.count)\n"
        summary += "- Columns: \(df.columnNames.joined(separator: ", "))\n"
        return summary
    }
}

/// Agentic Swift Execution Sandbox Evaluator.
public actor SwiftAgentEvaluator {
    /// Creates a new instance.
    public init() {}

    /// Evaluates dynamic DataFrame transformation expressions in a sandboxed environment.
    public func evaluate(command: String, on df: DataFrame) async throws -> DataFrame {
        let parsed = try parseCommand(command)
        do {
            switch parsed {
            case .filter(let column, let condition):
                return try df.filter(column: column, where: condition)
            case .sample(let n):
                return df.sample(n: Swift.min(df.rowCount, n))
            case .select(let columns):
                return try df.select(columns)
            case .head(let n):
                return df.head(n)
            case .tail(let n):
                return df.tail(n)
            }
        } catch {
            throw AgentError.executionFailed(error.localizedDescription)
        }
    }

    /// Parses a string command into a structured `AgentCommand` enum.
    public func parseCommand(_ command: String) throws -> AgentCommand {
        let trimmed = command.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw AgentError.unparseable(command) }

        let lower = trimmed.lowercased()

        // 1. filter command
        if lower.hasPrefix("filter") {
            let body = trimmed.dropFirst("filter".count).trimmingCharacters(in: .whitespacesAndNewlines)
            
            // Special cases: isNull, isNotNull
            if body.lowercased().hasSuffix("isnull") {
                let col = body.dropLast("isnull".count).trimmingCharacters(in: .whitespacesAndNewlines)
                guard !col.isEmpty else { throw AgentError.unparseable(command) }
                return .filter(column: col, condition: .isNull)
            }
            if body.lowercased().hasSuffix("isnotnull") {
                let col = body.dropLast("isnotnull".count).trimmingCharacters(in: .whitespacesAndNewlines)
                guard !col.isEmpty else { throw AgentError.unparseable(command) }
                return .filter(column: col, condition: .isNotNull)
            }

            let operators = [">=", "<=", "==", "!=", "=", ">", "<"]
            for opStr in operators {
                if let opRange = body.range(of: opStr) {
                    let col = String(body[..<opRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                    let valStr = String(body[opRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !col.isEmpty, !valStr.isEmpty else { continue }

                    let rhs: any Sendable
                    if let d = Double(valStr) {
                        rhs = d
                    } else {
                        rhs = valStr.trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
                    }

                    let cond: FilterCondition
                    switch opStr {
                    case ">=": cond = .greaterThanOrEqual(rhs)
                    case "<=": cond = .lessThanOrEqual(rhs)
                    case "==", "=": cond = .equals(rhs)
                    case "!=": cond = .notEquals(rhs)
                    case ">": cond = .greaterThan(rhs)
                    case "<": cond = .lessThan(rhs)
                    default: continue
                    }

                    return .filter(column: col, condition: cond)
                }
            }
            throw AgentError.unparseable(command)
        }

        // 2. sample command
        if lower.hasPrefix("sample") {
            let body = trimmed.dropFirst("sample".count).trimmingCharacters(in: .whitespacesAndNewlines)
            let cleaned = body.replacingOccurrences(of: "n=", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            let n = Int(cleaned) ?? 5
            guard n > 0 else { throw AgentError.unparseable(command) }
            return .sample(n: n)
        }

        // 3. select command
        if lower.hasPrefix("select") {
            let body = trimmed.dropFirst("select".count).trimmingCharacters(in: .whitespacesAndNewlines)
            let cols = body.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
            guard !cols.isEmpty else { throw AgentError.unparseable(command) }
            return .select(columns: cols)
        }

        // 4. head command
        if lower.hasPrefix("head") {
            let body = trimmed.dropFirst("head".count).trimmingCharacters(in: .whitespacesAndNewlines)
            let cleaned = body.replacingOccurrences(of: "n=", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            let n = Int(cleaned) ?? 5
            guard n >= 0 else { throw AgentError.unparseable(command) }
            return .head(n: n)
        }

        // 5. tail command
        if lower.hasPrefix("tail") {
            let body = trimmed.dropFirst("tail".count).trimmingCharacters(in: .whitespacesAndNewlines)
            let cleaned = body.replacingOccurrences(of: "n=", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            let n = Int(cleaned) ?? 5
            guard n >= 0 else { throw AgentError.unparseable(command) }
            return .tail(n: n)
        }

        throw AgentError.unparseable(command)
    }
}
