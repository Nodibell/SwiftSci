import Foundation
import SwiftDataFrame
import SwiftML

// MARK: - Leakage Category

/// The specific mechanism or pattern of target leakage detected.
public enum LeakageType: String, Sendable, Codable, CaseIterable {
    /// Feature exhibits an identical or near-perfect linear correlation with the target ($|r| \ge \text{threshold}$).
    case highLinearCorrelation
    /// Feature exhibits an identical or near-perfect monotonic rank correlation with the target.
    case highRankCorrelation
    /// Feature acts as an identifier, primary key, or monotonic row index that memorizes training order.
    case identifierLeakage
    /// Feature is an exact numerical duplicate of the target vector.
    case duplicateTarget
}

// MARK: - Leakage Violation Record

/// Diagnostic record of a specific feature flagged for target leakage.
public struct LeakageViolation: Sendable, Codable, Equatable {
    /// Identifier or column name of the flagged feature.
    public let featureName: String
    /// Classification of the detected leakage mechanism.
    public let type: LeakageType
    /// Quantitative strength of the association (e.g., Pearson $r$, Spearman $\rho$, or overlap ratio).
    public let score: Double
    /// Human-readable explanation and diagnostic context.
    public let detail: String
    
    /// Initializes a leakage violation record.
    /// - Parameters:
    ///   - featureName: Name of the feature.
    ///   - type: Category of leakage detected.
    ///   - score: Empirical association score.
    ///   - detail: Explanatory diagnostic message.
    public init(featureName: String, type: LeakageType, score: Double, detail: String) {
        self.featureName = featureName
        self.type = type
        self.score = score
        self.detail = detail
    }
}

// MARK: - Leakage Audit Report

/// Comprehensive pre-training audit report identifying target leakage risks and recommended feature exclusions.
public struct LeakageReport: Sendable, Codable, Equatable {
    /// All detected leakage violations across audited candidate features.
    public let violations: [LeakageViolation]
    /// Indicates whether critical leakage was discovered that would corrupt model generalisation.
    public let hasSevereLeakage: Bool
    /// Recommended list of feature names that should be excluded prior to model fitting.
    public let recommendedExclusions: [String]
    
    /// Initializes a leakage audit report.
    /// - Parameters:
    ///   - violations: List of detected violations.
    ///   - hasSevereLeakage: Flag indicating presence of critical leakage.
    ///   - recommendedExclusions: Array of feature names to drop before training.
    public init(violations: [LeakageViolation], hasSevereLeakage: Bool, recommendedExclusions: [String]) {
        self.violations = violations
        self.hasSevereLeakage = hasSevereLeakage
        self.recommendedExclusions = recommendedExclusions
    }
}

// MARK: - Target Leakage Detector

/// Pre-training guardrail sentry that audits feature-target relationships to prevent false-positive model performance.
public enum TargetLeakageDetector {
    /// Audits a 2D feature matrix against a 1D target vector.
    ///
    /// - Parameters:
    ///   - features: 2D array of continuous feature values of shape `[N, P]`.
    ///   - featureNames: Column names corresponding to the $P$ features.
    ///   - targets: 1D array of continuous or binary target values of length $N$.
    ///   - correlationThreshold: Absolute Pearson/Spearman correlation threshold for flagging (defaults to 0.999).
    /// - Returns: A `LeakageReport` detailing flagged violations and suggested feature exclusions.
    /// - Throws: `SwiftMLError.emptyInput` if data is empty, or `SwiftMLError.dimensionMismatch` if lengths differ.
    public static func audit(
        features: [[Double]],
        featureNames: [String],
        targets: [Double],
        correlationThreshold: Double = 0.999
    ) throws -> LeakageReport {
        guard !features.isEmpty && !targets.isEmpty else {
            throw SwiftMLError.emptyInput
        }
        let n = features.count
        guard n == targets.count else {
            throw SwiftMLError.dimensionMismatch(expected: n, got: targets.count)
        }
        let p = features[0].count
        guard p == featureNames.count else {
            throw SwiftMLError.invalidParameter("featureNames count (\(featureNames.count)) must match feature columns count (\(p))")
        }
        guard n >= 2 else {
            return LeakageReport(violations: [], hasSevereLeakage: false, recommendedExclusions: [])
        }
        
        var columns: [[Double]] = []
        columns.reserveCapacity(p)
        for colIdx in 0..<p {
            var col = [Double](repeating: 0.0, count: n)
            for rowIdx in 0..<n {
                col[rowIdx] = features[rowIdx][colIdx]
            }
            columns.append(col)
        }
        
        return auditColumns(columns: columns, featureNames: featureNames, targets: targets, threshold: correlationThreshold)
    }
    
    /// Audits DataFrame columns against a specified continuous or categorical target column.
    ///
    /// - Parameters:
    ///   - dataFrame: Input DataFrame containing features and target column.
    ///   - targetColumn: Name of the target column in the DataFrame.
    ///   - correlationThreshold: Absolute correlation threshold for flagging (defaults to 0.999).
    /// - Returns: A `LeakageReport` detailing flagged violations and suggested feature exclusions.
    /// - Throws: `SwiftMLError` if column is missing or types cannot be extracted.
    public static func audit(
        dataFrame: DataFrame,
        targetColumn: String,
        correlationThreshold: Double = 0.999
    ) throws -> LeakageReport {
        guard let targetCol = dataFrame[column: targetColumn] else {
            throw SwiftMLError.columnNotFound(targetColumn)
        }
        
        let n = dataFrame.rowCount
        guard n >= 2 else {
            return LeakageReport(violations: [], hasSevereLeakage: false, recommendedExclusions: [])
        }
        
        var targets = [Double](repeating: 0.0, count: n)
        if let dblCol = targetCol as? TypedColumn<Double> {
            for i in 0..<n { targets[i] = dblCol.values[i] ?? 0.0 }
        } else if let intCol = targetCol as? TypedColumn<Int> {
            for i in 0..<n { targets[i] = Double(intCol.values[i] ?? 0) }
        } else {
            throw SwiftMLError.invalidParameter("Target column '\(targetColumn)' must be numeric (Double or Int)")
        }
        
        var featureNames: [String] = []
        var columns: [[Double]] = []
        
        for col in dataFrame.columns {
            if col.name == targetColumn { continue }
            if let dblCol = col as? TypedColumn<Double> {
                featureNames.append(col.name)
                var vals = [Double](repeating: 0.0, count: n)
                for i in 0..<n { vals[i] = dblCol.values[i] ?? 0.0 }
                columns.append(vals)
            } else if let intCol = col as? TypedColumn<Int> {
                featureNames.append(col.name)
                var vals = [Double](repeating: 0.0, count: n)
                for i in 0..<n { vals[i] = Double(intCol.values[i] ?? 0) }
                columns.append(vals)
            }
        }
        
        return auditColumns(columns: columns, featureNames: featureNames, targets: targets, threshold: correlationThreshold)
    }
    
    // MARK: - Core Audit Pipeline
    
    private static func auditColumns(
        columns: [[Double]],
        featureNames: [String],
        targets: [Double],
        threshold: Double
    ) -> LeakageReport {
        let n = targets.count
        var violations: [LeakageViolation] = []
        var exclusions: Set<String> = []
        
        let targetRanks = computeRanks(targets)
        let idKeywords = ["id", "uuid", "row_id", "rowid", "index", "key", "pk", "applicant_id", "user_id"]
        
        for colIdx in 0..<columns.count {
            let colName = featureNames[colIdx]
            let colVals = columns[colIdx]
            
            // 1. Duplicate Target Check
            var isDuplicate = true
            for i in 0..<n {
                if abs(colVals[i] - targets[i]) > 1e-10 {
                    isDuplicate = false
                    break
                }
            }
            if isDuplicate {
                violations.append(LeakageViolation(
                    featureName: colName,
                    type: .duplicateTarget,
                    score: 1.0,
                    detail: "Feature '\(colName)' is an exact duplicate of the target label vector."
                ))
                exclusions.insert(colName)
                continue
            }
            
            // 2. Pearson Linear Correlation Check
            let r = pearsonCorrelation(colVals, targets)
            let absR = abs(r)
            if absR >= threshold && !absR.isNaN {
                violations.append(LeakageViolation(
                    featureName: colName,
                    type: .highLinearCorrelation,
                    score: absR,
                    detail: "Feature '\(colName)' exhibits extreme linear correlation with target (|r| = \(absR) >= \(threshold))."
                ))
                exclusions.insert(colName)
            }
            
            // 3. Spearman Rank Correlation Check
            let colRanks = computeRanks(colVals)
            let rho = pearsonCorrelation(colRanks, targetRanks)
            let absRho = abs(rho)
            if absRho >= threshold && absR < threshold && !absRho.isNaN {
                violations.append(LeakageViolation(
                    featureName: colName,
                    type: .highRankCorrelation,
                    score: absRho,
                    detail: "Feature '\(colName)' exhibits extreme monotonic rank correlation with target (|ρ| = \(absRho) >= \(threshold))."
                ))
                exclusions.insert(colName)
            }
            
            // 4. Identifier / Sequential Index Leakage Check
            let nameLower = colName.lowercased()
            var isIDName = false
            for kw in idKeywords {
                if nameLower == kw || nameLower.hasSuffix("_\(kw)") {
                    isIDName = true
                    break
                }
            }
            let isMonotonicSequential = checkMonotonicSequential(colVals)
            let uniqueCount = Set(colVals).count
            let uniquenessRatio = Double(uniqueCount) / Double(n)
            
            if (isIDName || isMonotonicSequential) && uniquenessRatio > 0.95 && (absR > 0.5 || absRho > 0.5) {
                violations.append(LeakageViolation(
                    featureName: colName,
                    type: .identifierLeakage,
                    score: max(absR, absRho),
                    detail: "Feature '\(colName)' exhibits primary key/identifier characteristics memorizing target ordering."
                ))
                exclusions.insert(colName)
            }
        }
        
        let hasSevere = !exclusions.isEmpty
        let orderedExclusions = featureNames.filter { exclusions.contains($0) }
        
        return LeakageReport(
            violations: violations,
            hasSevereLeakage: hasSevere,
            recommendedExclusions: orderedExclusions
        )
    }
    
    // MARK: - Internal Math Helpers
    
    private static func pearsonCorrelation(_ x: [Double], _ y: [Double]) -> Double {
        let n = Double(x.count)
        guard n >= 2 else { return 0.0 }
        
        var sumX = 0.0
        var sumY = 0.0
        for i in 0..<x.count {
            sumX += x[i]
            sumY += y[i]
        }
        let meanX = sumX / n
        let meanY = sumY / n
        
        var cov = 0.0
        var varX = 0.0
        var varY = 0.0
        for i in 0..<x.count {
            let dx = x[i] - meanX
            let dy = y[i] - meanY
            cov += dx * dy
            varX += dx * dx
            varY += dy * dy
        }
        
        let denom = sqrt(varX * varY)
        guard denom > 1e-12 else { return 0.0 }
        return cov / denom
    }
    
    private static func computeRanks(_ values: [Double]) -> [Double] {
        let n = values.count
        let indexed = values.enumerated().sorted(by: { $0.element < $1.element })
        var ranks = [Double](repeating: 0.0, count: n)
        
        var i = 0
        while i < n {
            var j = i
            while j < n - 1 && abs(indexed[j + 1].element - indexed[j].element) < 1e-12 {
                j += 1
            }
            let avgRank = Double(i + j + 2) / 2.0
            for k in i...j {
                ranks[indexed[k].offset] = avgRank
            }
            i = j + 1
        }
        return ranks
    }
    
    private static func checkMonotonicSequential(_ values: [Double]) -> Bool {
        guard values.count >= 3 else { return false }
        let step = values[1] - values[0]
        guard abs(abs(step) - 1.0) < 1e-6 else { return false }
        for i in 2..<values.count {
            if abs((values[i] - values[i - 1]) - step) > 1e-6 {
                return false
            }
        }
        return true
    }
}
