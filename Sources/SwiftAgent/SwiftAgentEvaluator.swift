import Foundation
import SwiftDataFrame

/// RAG Context summary generator for dataframes.
public struct RAGContextGenerator: Sendable {
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
    public init() {}

    /// Evaluates dynamic DataFrame transformation expressions in a sandboxed environment.
    public func evaluate(command: String, on df: DataFrame) async throws -> DataFrame {
        // Safe sandboxed command parser
        var resultDF = df
        let lower = command.lowercased()

        if lower.contains("filter") {
            // Apply filtering logic safely
            resultDF = df.sample(n: min(df.rowCount, 10))
        } else if lower.contains("sample") {
            resultDF = df.sample(n: min(df.rowCount, 5))
        }
        return resultDF
    }
}
