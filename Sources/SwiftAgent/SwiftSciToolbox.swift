import Foundation
import SwiftDataFrame
import SwiftStats
import SwiftPreprocessing
import SwiftML
import SwiftCluster
import SwiftForecast
import SwiftExplain
import SwiftNLP
import SwiftDatabase
import SwiftVisualization

/// A comprehensive suite of pre-built, sandboxed agent tools exposing the full SwiftSci ecosystem to autonomous agents.
public enum SwiftSciToolbox {

    // MARK: - 1. Statistical Hypothesis & Summary Tool

    /// Agent tool for computing descriptive statistics and hypothesis testing (t-test, correlation) on tabular columns.
    public struct StatisticsInspectionTool: StructuredAgentTool, Sendable {
        /// The unique tool identifier.
        public let name: String = "StatisticsInspection"
        /// Human-readable explanation for model tool selection.
        public let description: String = "Computes summary statistics, Welch's two-sample t-test, or Pearson/Spearman correlation between numerical columns."
        /// Parameter schema definition.
        public let parameterSchema: AgentParameterSchema = AgentParameterSchema(
            properties: [
                "action": AgentParameterProperty(type: "string", description: "Operation: 'summary', 'ttest', or 'correlation'.", enum: ["summary", "ttest", "correlation"]),
                "columnA": AgentParameterProperty(type: "string", description: "Name of the primary column or comma-separated numbers."),
                "columnB": AgentParameterProperty(type: "string", description: "Name of the secondary column for t-test or correlation.")
            ],
            required: ["action", "columnA"]
        )

        private let dataframeProvider: (@Sendable () -> DataFrame?)?

        /// Initializes the statistics tool.
        /// - Parameter dataframeProvider: Optional closure providing the active DataFrame.
        public init(dataframeProvider: (@Sendable () -> DataFrame?)? = nil) {
            self.dataframeProvider = dataframeProvider
        }

        /// Executes the structured statistics inspection.
        public func executeStructured(arguments: [String: String]) async throws -> AgentToolOutput {
            let start = CFAbsoluteTimeGetCurrent()
            let action = arguments["action"] ?? "summary"
            let colAName = arguments["columnA"] ?? ""
            let colBName = arguments["columnB"] ?? ""

            let df = dataframeProvider?()
            func extractValues(_ spec: String) -> [Double] {
                if let df = df, let col = df[column: spec] {
                    return col.toDoubles() ?? []
                }
                return spec.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            }

            let valsA = extractValues(colAName)
            guard !valsA.isEmpty else {
                throw AgentError.executionFailed("No numeric values found for primary column: '\(colAName)'.")
            }

            switch action.lowercased() {
            case "ttest":
                let valsB = extractValues(colBName)
                guard !valsB.isEmpty else {
                    throw AgentError.executionFailed("Welch's t-test requires valid secondary column 'columnB'.")
                }
                let tRes = try Stats.tTest(sample1: valsA, sample2: valsB, equalVariances: false)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
                let text = "Welch's Two-Sample t-Test:\n- t-statistic: \(String(format: "%.4f", tRes.statistic))\n- p-value: \(String(format: "%.6e", tRes.pValue))\n- Degrees of Freedom: \(String(format: "%.2f", tRes.degreesOfFreedom))\n- Significant (α=0.05): \(tRes.isSignificant)"
                return AgentToolOutput(text: text, durationMs: elapsed)

            case "correlation":
                let valsB = extractValues(colBName)
                guard !valsB.isEmpty else {
                    throw AgentError.executionFailed("Correlation requires valid secondary column 'columnB'.")
                }
                let pearson = try Stats.pearsonCorrelation(valsA, valsB)
                let spearman = try Stats.spearmanCorrelation(valsA, valsB)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
                let text = "Bivariate Correlation:\n- Pearson r: \(String(format: "%.4f", pearson))\n- Spearman ρ: \(String(format: "%.4f", spearman))"
                return AgentToolOutput(text: text, durationMs: elapsed)

            default:
                let m = try Stats.mean(valsA)
                let s = try Stats.standardDeviation(valsA)
                let v = try Stats.variance(valsA)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
                let text = "Descriptive Statistics for '\(colAName)':\n- Count: \(valsA.count)\n- Mean: \(String(format: "%.4f", m))\n- StdDev: \(String(format: "%.4f", s))\n- Variance: \(String(format: "%.4f", v))"
                return AgentToolOutput(text: text, durationMs: elapsed)
            }
        }
    }

    // MARK: - 2. Data Drift & Distribution Shift Tool

    /// Agent tool evaluating statistical data drift using Wasserstein Distance W1 and Population Stability Index (PSI).
    public struct DataDriftInspectionTool: StructuredAgentTool, Sendable {
        /// The unique tool identifier.
        public let name: String = "DataDriftInspection"
        /// Description of drift detection capabilities.
        public let description: String = "Monitors distribution shift and data drift between baseline reference data and production data using Wasserstein W1 distance and PSI."
        /// Parameter schema definition.
        public let parameterSchema: AgentParameterSchema = AgentParameterSchema(
            properties: [
                "reference": AgentParameterProperty(type: "string", description: "Comma-separated baseline numbers or reference column name."),
                "production": AgentParameterProperty(type: "string", description: "Comma-separated production numbers or production column name.")
            ],
            required: ["reference", "production"]
        )

        /// Initializes data drift tool.
        public init() {}

        /// Evaluates data drift metrics between two distributions.
        public func executeStructured(arguments: [String: String]) async throws -> AgentToolOutput {
            let start = CFAbsoluteTimeGetCurrent()
            guard let refStr = arguments["reference"], let prodStr = arguments["production"] else {
                throw AgentError.executionFailed("Both 'reference' and 'production' parameters are required.")
            }

            let ref = refStr.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            let prod = prodStr.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }

            guard !ref.isEmpty && !prod.isEmpty else {
                throw AgentError.executionFailed("Could not parse numeric distributions for reference or production.")
            }

            let w1 = try WassersteinDistance.compute(ref, prod)
            let psiRes = try PopulationStabilityIndex.compute(expected: ref, actual: prod)
            let psi = psiRes.psi

            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
            let alertText: String
            if psi >= 0.25 {
                alertText = "🚨 CRITICAL DRIFT DETECTED (PSI >= 0.25)"
            } else if psi >= 0.10 {
                alertText = "⚠️ MODERATE DRIFT DETECTED (0.10 <= PSI < 0.25)"
            } else {
                alertText = "✅ STABLE DISTRIBUTION (PSI < 0.10)"
            }

            let text = "Data Drift Assessment:\n- Status: \(alertText)\n- Population Stability Index (PSI): \(String(format: "%.4f", psi))\n- Wasserstein Distance (W1): \(String(format: "%.4f", w1))"
            return AgentToolOutput(text: text, durationMs: elapsed)
        }
    }

    // MARK: - 3. Time Series Forecasting Tool

    /// Agent tool generating forecasts with analytical confidence intervals using AutoARIMA.
    public struct TimeSeriesForecastingTool: StructuredAgentTool, Sendable {
        /// The unique tool identifier.
        public let name: String = "TimeSeriesForecasting"
        /// Description of forecasting capabilities.
        public let description: String = "Generates multi-step ahead forecasts for time series data using ARIMA."
        /// Parameter schema definition.
        public let parameterSchema: AgentParameterSchema = AgentParameterSchema(
            properties: [
                "values": AgentParameterProperty(type: "string", description: "Comma-separated historical values array."),
                "horizon": AgentParameterProperty(type: "integer", description: "Forecast horizon (steps ahead). Default: 12.")
            ],
            required: ["values"]
        )

        /// Initializes forecasting tool.
        public init() {}

        /// Generates time series extrapolation.
        public func executeStructured(arguments: [String: String]) async throws -> AgentToolOutput {
            let start = CFAbsoluteTimeGetCurrent()
            guard let valStr = arguments["values"] else {
                throw AgentError.executionFailed("Parameter 'values' is required.")
            }
            let values = valStr.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            guard values.count >= 10 else {
                throw AgentError.executionFailed("Time series forecasting requires at least 10 historical observations.")
            }
            let h = Int(arguments["horizon"] ?? "12") ?? 12

            let arima = try ARIMA(p: 1, d: 1, q: 1)
            try await arima.fit(series: values)
            let res = try await arima.forecast(horizon: h)

            let formatted = res.forecast.predictions.map { String(format: "%.2f", $0) }.joined(separator: ", ")
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
            let text = "ARIMA(1,1,1) Forecast (h=\(h)):\n[\(formatted)]"
            return AgentToolOutput(text: text, durationMs: elapsed)
        }
    }

    // MARK: - 4. NLP Sentiment & Text Analysis Tool

    /// Agent tool for linguistic sentiment analysis via VADER.
    public struct NLPSentimentTool: StructuredAgentTool, Sendable {
        /// The unique tool identifier.
        public let name: String = "NLPSentiment"
        /// Description of sentiment tool.
        public let description: String = "Evaluates sentiment polarity (compound, positive, neutral, negative) of textual statements using VADER."
        /// Parameter schema definition.
        public let parameterSchema: AgentParameterSchema = AgentParameterSchema(
            properties: [
                "text": AgentParameterProperty(type: "string", description: "Text passage to analyze.")
            ],
            required: ["text"]
        )

        /// Initializes sentiment tool.
        public init() {}

        /// Executes sentiment analysis.
        public func executeStructured(arguments: [String: String]) async throws -> AgentToolOutput {
            let start = CFAbsoluteTimeGetCurrent()
            guard let text = arguments["text"] else {
                throw AgentError.executionFailed("Parameter 'text' is required.")
            }
            let vader = VADERSentimentAnalyzer()
            let score = vader.polarityScores(text: text)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0

            let mood = score.compound >= 0.05 ? "Positive 🟢" : (score.compound <= -0.05 ? "Negative 🔴" : "Neutral ⚪")
            let out = "VADER Sentiment Analysis:\n- Classification: \(mood)\n- Compound Score: \(String(format: "%.4f", score.compound))\n- Positive: \(String(format: "%.3f", score.pos))\n- Neutral: \(String(format: "%.3f", score.neu))\n- Negative: \(String(format: "%.3f", score.neg))"
            return AgentToolOutput(text: out, durationMs: elapsed)
        }
    }

    // MARK: - 5. Safe Sandboxed SQL Database Query Tool

    /// Agent tool executing read-only SQL queries on SQLite databases with safety sentry blocking mutation statements.
    public struct SafeDatabaseQueryTool: StructuredAgentTool, Sendable {
        /// The unique tool identifier.
        public let name: String = "DatabaseQuery"
        /// Description of SQL query capability.
        public let description: String = "Executes read-only SQL SELECT queries on the active relational database. Mutation commands (DROP, DELETE, UPDATE, INSERT) are blocked."
        /// Indicates that queries require explicit approval if mutating keywords are suspected.
        public let requiresApproval: Bool = false
        /// Parameter schema definition.
        public let parameterSchema: AgentParameterSchema = AgentParameterSchema(
            properties: [
                "query": AgentParameterProperty(type: "string", description: "The SQL SELECT query string to execute.")
            ],
            required: ["query"]
        )

        private let databasePath: String

        /// Initializes the database tool for a given SQLite file path.
        /// - Parameter databasePath: Path to SQLite database file.
        public init(databasePath: String = ":memory:") {
            self.databasePath = databasePath
        }

        /// Executes read-only SQL query with statement safety validation.
        public func executeStructured(arguments: [String: String]) async throws -> AgentToolOutput {
            let start = CFAbsoluteTimeGetCurrent()
            guard let query = arguments["query"]?.trimmingCharacters(in: .whitespacesAndNewlines) else {
                throw AgentError.executionFailed("Parameter 'query' is required.")
            }

            // Sentry: Enforce read-only semantics
            let blockedKeywords = ["drop", "delete", "update", "insert", "alter", "truncate", "create"]
            let lowerQuery = query.lowercased()
            for keyword in blockedKeywords {
                if lowerQuery.contains(keyword) {
                    throw AgentError.executionFailed("Security Violation: Disallowed mutation keyword '\(keyword.uppercased())' in query. Only read-only SELECT statements are permitted.")
                }
            }

            let connection = SQLiteConnection(databasePath: databasePath)
            let result = try await connection.executeQuery(query)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0

            var out = "Query executed successfully (\(result.rows.count) rows returned):\n"
            out += "- Columns: [\(result.columns.joined(separator: ", "))]\n"
            for (idx, row) in result.rows.prefix(3).enumerated() {
                let formattedRow = row.map { "\($0)" }.joined(separator: ", ")
                out += "- Row \(idx + 1): [\(formattedRow)]\n"
            }
            if result.rows.count > 3 {
                out += "- ... (\(result.rows.count - 3) more rows)\n"
            }
            return AgentToolOutput(text: out, durationMs: elapsed)
        }
    }

    // MARK: - Factory

    /// Creates a complete default toolbox of scientific and analytical tools.
    ///
    /// - Parameters:
    ///   - dataframeProvider: Optional provider closure returning the current active DataFrame.
    ///   - databasePath: Path to active SQLite database (default: `:memory:`).
    /// - Returns: An array of ready-to-use structured agent tools.
    public static func standardTools(
        dataframeProvider: (@Sendable () -> DataFrame?)? = nil,
        databasePath: String = ":memory:"
    ) -> [any AgentTool] {
        return [
            StatisticsInspectionTool(dataframeProvider: dataframeProvider),
            DataDriftInspectionTool(),
            TimeSeriesForecastingTool(),
            NLPSentimentTool(),
            SafeDatabaseQueryTool(databasePath: databasePath)
        ]
    }
}
