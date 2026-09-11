import Foundation
import SwiftDataFrame

/// Errors thrown by SwiftAgent evaluation sandbox.
public enum AgentError: Error, LocalizedError, Equatable {
    case unparseable(String)
    case executionFailed(String)
    case toolTimeout(tool: String, seconds: Double)

    /// The error description.
    public var errorDescription: String? {
        switch self {
        case .unparseable(let cmd):
            return "Failed to parse agent command: '\(cmd)'."
        case .executionFailed(let msg):
            return "Agent execution failed: \(msg)"
        case .toolTimeout(let tool, let seconds):
            return "Tool '\(tool)' execution timed out after \(seconds) seconds."
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
    /// - Parameters:
    ///   - df: Input DataFrame instance.
    ///   - name: Name or identifier string.
    /// - Returns: Generated or formatted text string.
    public func generateSummary(df: DataFrame, name: String = "Dataset") -> String {
        var summary = "## \(name) Profile\n"
        summary += "- Rows: \(df.rowCount), Columns: \(df.columnNames.count)\n"
        summary += "- Columns: \(df.columnNames.joined(separator: ", "))\n"
        return summary
    }
}

/// Represents an immutable audit trail entry recording a discrete data transformation step.
public struct LineageRecord: Sendable, Codable, Equatable {
    /// Sequential index of the execution step within the session.
    public let stepIndex: Int
    /// DSL operation executed (e.g. "filter", "groupBy", "fillNulls").
    public let operation: String
    /// Row count of the dataset prior to the transformation.
    public let inputRows: Int
    /// Row count of the dataset following the transformation.
    public let outputRows: Int
    /// Timestamp when the transformation concluded.
    public let timestamp: Date

    /// Creates a new lineage audit record.
    /// - Parameters:
    ///   - stepIndex: Sequential execution index.
    ///   - operation: Description of the transformation operation.
    ///   - inputRows: Number of rows in the input dataset.
    ///   - outputRows: Number of rows in the resulting dataset.
    ///   - timestamp: Timestamp when the step was executed.
    public init(
        stepIndex: Int,
        operation: String,
        inputRows: Int,
        outputRows: Int,
        timestamp: Date = Date()
    ) {
        self.stepIndex = stepIndex
        self.operation = operation
        self.inputRows = inputRows
        self.outputRows = outputRows
        self.timestamp = timestamp
    }
}

/// Agentic Swift Execution Sandbox Evaluator.
public actor SwiftAgentEvaluator {
    /// Audit trail of transformations executed by this evaluator instance.
    public private(set) var lineage: [LineageRecord] = []

    /// Creates a new instance.
    public init() {}

    /// Clears the recorded lineage audit trail.
    public func clearLineage() {
        lineage.removeAll()
    }

    /// Evaluates dynamic DataFrame transformation expressions in a sandboxed environment.
    /// - Parameters:
    ///   - command: String DSL command (e.g. "filter age > 30", "head 5", "dropNulls").
    ///   - df: The input DataFrame.
    /// - Returns: The transformed DataFrame.
    /// - Throws: `AgentError` if command parsing or execution fails.
    ///
    /// ## Lineage Tracking
    /// Every successful command appends an immutable `LineageRecord` to the evaluator's `lineage` history.
    public func evaluate(command: String, on df: DataFrame) async throws -> DataFrame {
        let parsed = try parseCommand(command)
        let inputRows = df.rowCount
        let opName: String
        let result: DataFrame
        do {
            switch parsed {
            case .filter(let column, let condition):
                opName = "filter(\(column))"
                result = try df.filter(column: column, where: condition)
            case .sample(let n):
                opName = "sample(\(n))"
                result = df.sample(n: Swift.min(df.rowCount, n))
            case .select(let columns):
                opName = "select(\(columns.joined(separator: ",")))"
                result = try df.select(columns)
            case .head(let n):
                opName = "head(\(n))"
                result = df.head(n)
            case .tail(let n):
                opName = "tail(\(n))"
                result = df.tail(n)
            case .rename(let from, let to):
                opName = "rename(\(from)->\(to))"
                result = try df.renameColumn(from, to: to)
            case .dropNulls(let columns):
                opName = "dropNulls"
                result = try dropNullRows(in: df, columns: columns)
            case .fillNulls(let column, let value):
                opName = "fillNulls(\(column))"
                guard let col = df[column: column, as: Double.self] else {
                    throw AgentError.executionFailed("Column '\(column)' not found or not a Double column")
                }
                let filled = col.fillNull(with: value)
                result = try df.withColumn(column, column: filled)
            case .groupBy(let column, let aggregation, let targetColumn):
                opName = "groupBy(\(column),\(aggregation))"
                let grouped = df.groupBy(column)
                if let targetColumn {
                    result = grouped.agg([targetColumn: aggregation])
                } else {
                    switch aggregation {
                    case .sum: result = grouped.sum()
                    case .mean: result = grouped.mean()
                    case .min: result = grouped.min()
                    case .max: result = grouped.max()
                    case .count: result = grouped.count()
                    case .first, .last:
                        throw AgentError.executionFailed("Aggregation '\(aggregation)' requires an explicit target column")
                    }
                }
            }
        } catch let error as AgentError {
            throw error
        } catch {
            throw AgentError.executionFailed(error.localizedDescription)
        }

        let record = LineageRecord(
            stepIndex: lineage.count + 1,
            operation: opName,
            inputRows: inputRows,
            outputRows: result.rowCount
        )
        lineage.append(record)
        return result
    }

    /// Parses a string command into a structured `AgentCommand` enum.
    /// - Parameters:
    ///   - command: Database SQL statement or terminal command.
    /// - Throws: `AgentError` or `SwiftMLError` if tool execution, AST evaluation, or reasoning fails.
    /// - Returns: The computed AgentCommand result instance.
    public func parseCommand(_ command: String) throws -> AgentCommand {
        let trimmed = command.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw AgentError.unparseable(command) }

        let lower = trimmed.lowercased()

        func cleanBody(_ prefix: String) -> String {
            var body = trimmed.dropFirst(prefix.count).trimmingCharacters(in: .whitespacesAndNewlines)
            if body.hasPrefix(":") {
                body = body.dropFirst().trimmingCharacters(in: .whitespacesAndNewlines)
            }
            if body.hasPrefix("(") && body.hasSuffix(")") {
                body = String(body.dropFirst().dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            return body
        }

        // 1. filter command
        if lower.hasPrefix("filter") {
            var body = cleanBody("filter")
            if body.contains("column:") && body.contains("condition:") {
                let parts = body.components(separatedBy: "condition:")
                let colPart = parts[0].replacingOccurrences(of: "column:", with: "")
                    .replacingOccurrences(of: "\"", with: "")
                    .replacingOccurrences(of: "'", with: "")
                    .replacingOccurrences(of: ",", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let condPart = parts[1].replacingOccurrences(of: "\"", with: "")
                    .replacingOccurrences(of: "'", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                body = "\(colPart) \(condPart)"
            }
            
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
            let body = cleanBody("sample")
            let cleaned = body.replacingOccurrences(of: "n=", with: "").replacingOccurrences(of: "n:", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            let n = Int(cleaned) ?? 5
            guard n > 0 else { throw AgentError.unparseable(command) }
            return .sample(n: n)
        }

        // 3. select command
        if lower.hasPrefix("select") {
            var body = cleanBody("select")
            if body.contains("columns:") {
                body = body.replacingOccurrences(of: "columns:", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            }
            body = body.replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "").replacingOccurrences(of: "\"", with: "")
            let cols = body.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
            guard !cols.isEmpty else { throw AgentError.unparseable(command) }
            return .select(columns: cols)
        }

        // 4. head command
        if lower.hasPrefix("head") {
            let body = cleanBody("head")
            let cleaned = body.replacingOccurrences(of: "n=", with: "").replacingOccurrences(of: "n:", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            let n = Int(cleaned) ?? 5
            guard n >= 0 else { throw AgentError.unparseable(command) }
            return .head(n: n)
        }

        // 5. tail command
        if lower.hasPrefix("tail") {
            let body = cleanBody("tail")
            let cleaned = body.replacingOccurrences(of: "n=", with: "").replacingOccurrences(of: "n:", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            let n = Int(cleaned) ?? 5
            guard n >= 0 else { throw AgentError.unparseable(command) }
            return .tail(n: n)
        }

        // 6. rename command
        if lower.hasPrefix("rename") {
            let body = cleanBody("rename")
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
            let body = cleanBody("dropnulls")
            if body.isEmpty {
                return .dropNulls(columns: nil)
            }
            let cols = body.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
            guard !cols.isEmpty else { throw AgentError.unparseable(command) }
            return .dropNulls(columns: cols)
        }

        // 8. fillnulls command
        if lower.hasPrefix("fillnulls") {
            let body = cleanBody("fillnulls")
            let parts = body.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
            guard parts.count == 2, let value = Double(parts[1]) else {
                throw AgentError.unparseable(command)
            }
            return .fillNulls(column: parts[0], value: value)
        }

        // 9. groupby command
        if lower.hasPrefix("groupby") {
            let body = cleanBody("groupby")
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
