// MARK: – Descriptive Statistics

/// Summary statistics for a numeric array.
public struct DescriptiveStats: Sendable, CustomStringConvertible {
    public let count: Int
    public let mean: Double
    public let standardDeviation: Double
    public let variance: Double
    public let min: Double
    public let q1: Double       // 25th percentile
    public let median: Double
    public let q3: Double       // 75th percentile
    public let max: Double
    public let skewness: Double
    public let kurtosis: Double
    public let nullCount: Int

    public var description: String {
        """
        count   \(count)
        mean    \(String(format: "%.6f", mean))
        std     \(String(format: "%.6f", standardDeviation))
        min     \(String(format: "%.6f", min))
        25%     \(String(format: "%.6f", q1))
        50%     \(String(format: "%.6f", median))
        75%     \(String(format: "%.6f", q3))
        max     \(String(format: "%.6f", max))
        skew    \(String(format: "%.6f", skewness))
        kurt    \(String(format: "%.6f", kurtosis))
        nulls   \(nullCount)
        """
    }
}

// MARK: – Hypothesis Tests

/// Result of a Student's t-test.
public struct TTestResult: Sendable, CustomStringConvertible {
    /// The t-statistic.
    public let statistic: Double
    /// Two-tailed p-value.
    public let pValue: Double
    /// Degrees of freedom.
    public let degreesOfFreedom: Double
    /// 95% confidence interval for the mean difference.
    public let confidenceInterval: ConfidenceInterval
    /// Cohen's d effect size.
    public let effectSize: Double

    /// Whether the result is significant at α = 0.05.
    public var isSignificant: Bool { pValue < 0.05 }

    public var description: String {
        "t(\(String(format: "%.2f", degreesOfFreedom))) = \(String(format: "%.4f", statistic)), p = \(String(format: "%.4f", pValue)), d = \(String(format: "%.4f", effectSize))"
    }
}

/// Result of a one-way ANOVA test.
public struct ANOVAResult: Sendable, CustomStringConvertible {
    /// The F-statistic.
    public let fStatistic: Double
    /// p-value from the F-distribution.
    public let pValue: Double
    /// Degrees of freedom (between groups): k - 1.
    public let dfBetween: Int
    /// Degrees of freedom (within groups): N - k.
    public let dfWithin: Int
    /// η² (eta-squared) effect size.
    public let etaSquared: Double

    public var isSignificant: Bool { pValue < 0.05 }

    public var description: String {
        "F(\(dfBetween), \(dfWithin)) = \(String(format: "%.4f", fStatistic)), p = \(String(format: "%.4f", pValue)), η² = \(String(format: "%.4f", etaSquared))"
    }
}

/// Result of a chi-square goodness-of-fit test.
public struct ChiSquareResult: Sendable, CustomStringConvertible {
    public let statistic: Double
    public let pValue: Double
    public let degreesOfFreedom: Int

    public var isSignificant: Bool { pValue < 0.05 }

    public var description: String {
        "χ²(\(degreesOfFreedom)) = \(String(format: "%.4f", statistic)), p = \(String(format: "%.4f", pValue))"
    }
}

/// Result of a normality test (Shapiro-Wilk or KS).
public struct NormalityTestResult: Sendable, CustomStringConvertible {
    public let statistic: Double
    public let pValue: Double
    /// True if p ≥ 0.05 (fail to reject normality at α = 0.05).
    public var isNormal: Bool { pValue >= 0.05 }

    public var description: String {
        "W = \(String(format: "%.4f", statistic)), p = \(String(format: "%.4f", pValue)) → \(isNormal ? "normal" : "not normal")"
    }
}

// MARK: – Support types

/// A symmetric confidence interval.
public struct ConfidenceInterval: Sendable, CustomStringConvertible {
    public let lower: Double
    public let upper: Double
    public let confidence: Double // e.g. 0.95

    public var description: String {
        "[\(String(format: "%.4f", lower)), \(String(format: "%.4f", upper))]"
    }
}

/// Norm order for vector norm computation.
public enum NormOrder: Sendable {
    case l1
    case l2
    case infinity
}
