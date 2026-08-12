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
    case rename(from: String, to: String)
    case dropNulls(columns: [String]?)
    case fillNulls(column: String, value: Double)
    case groupBy(column: String, aggregation: Aggregation, targetColumn: String?)
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
            case .rename(let from, let to):
                return try df.renameColumn(from, to: to)
            case .dropNulls(let columns):
                return try dropNullRows(in: df, columns: columns)
            case .fillNulls(let column, let value):
                guard let col = df[column: column, as: Double.self] else {
                    throw AgentError.executionFailed("Column '\(column)' not found or not a Double column")
                }
                let filled = col.fillNull(with: value)
                return try df.withColumn(column, column: filled)
            case .groupBy(let column, let aggregation, let targetColumn):
                let grouped = df.groupBy(column)
                if let targetColumn {
                    return grouped.agg([targetColumn: aggregation])
                }
                switch aggregation {
                case .sum: return grouped.sum()
                case .mean: return grouped.mean()
                case .min: return grouped.min()
                case .max: return grouped.max()
                case .count: return grouped.count()
                case .first, .last:
                    throw AgentError.executionFailed("Aggregation '\(aggregation)' requires an explicit target column")
                }
            }
        } catch let error as AgentError {
            throw error
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

        // 6. rename command
        if lower.hasPrefix("rename") {
            let body = trimmed.dropFirst("rename".count).trimmingCharacters(in: .whitespacesAndNewlines)
            guard let toRange = body.range(of: " to ", options: .caseInsensitive) else {
                throw AgentError.unparseable(command)
            }
            let from = String(body[..<toRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            let to = String(body[toRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !from.isEmpty, !to.isEmpty else { throw AgentError.unparseable(command) }
            return .rename(from: from, to: to)
        }

        // 7. dropnulls command
        if lower.hasPrefix("dropnulls") {
            let body = trimmed.dropFirst("dropnulls".count).trimmingCharacters(in: .whitespacesAndNewlines)
            if body.isEmpty {
                return .dropNulls(columns: nil)
            }
            let cols = body.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
            guard !cols.isEmpty else { throw AgentError.unparseable(command) }
            return .dropNulls(columns: cols)
        }

        // 8. fillnulls command
        if lower.hasPrefix("fillnulls") {
            let body = trimmed.dropFirst("fillnulls".count).trimmingCharacters(in: .whitespacesAndNewlines)
            let parts = body.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
            guard parts.count == 2, let value = Double(parts[1]) else {
                throw AgentError.unparseable(command)
            }
            return .fillNulls(column: parts[0], value: value)
        }

        // 9. groupby command
        if lower.hasPrefix("groupby") {
            let body = trimmed.dropFirst("groupby".count).trimmingCharacters(in: .whitespacesAndNewlines)
            let parts = body.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
            guard parts.count == 2 || parts.count == 3 else {
                throw AgentError.unparseable(command)
            }
            guard let aggregation = parseAggregation(parts[1]) else {
                throw AgentError.unparseable(command)
            }
            let targetColumn = parts.count == 3 ? parts[2] : nil
            return .groupBy(column: parts[0], aggregation: aggregation, targetColumn: targetColumn)
        }

        throw AgentError.unparseable(command)
    }

    private func dropNullRows(in df: DataFrame, columns: [String]?) throws -> DataFrame {
        let cols = columns ?? df.columnNames
        guard !cols.isEmpty else { return df }
        for col in cols {
            guard df[column: col] != nil else {
                throw AgentError.executionFailed("Column '\(col)' not found")
            }
        }
        let indices = (0..<df.rowCount).filter { i in
            !cols.contains { col in df[column: col]?.value(at: i) == nil }
        }
        return df.gathered(at: indices)
    }

    private func parseAggregation(_ token: String) -> Aggregation? {
        switch token.lowercased() {
        case "sum": return .sum
        case "mean": return .mean
        case "min": return .min
        case "max": return .max
        case "count": return .count
        case "first": return .first
        case "last": return .last
        default: return nil
        }
    }
}
