import Accelerate
import SwiftDataFrame

// MARK: – Descriptive Statistics (vDSP-backed)

extension Stats {

    /// Arithmetic mean using vDSP.mean.
    /// - Parameters:
    ///   - values: <#description#>
    ///   - checkNaN: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func mean(_ values: [Double], checkNaN: Bool = true) throws -> Double {
        try requireNonEmpty(values)
        let result = vDSP.mean(values)
        if checkNaN && result.isNaN { try requireNoNaN(values) }
        return result
    }

    /// Float overload.
    /// - Parameters:
    ///   - values: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func mean(_ values: [Float]) throws -> Float {
        guard !values.isEmpty else { throw StatsError.emptyInput }
        return vDSP.mean(values)
    }

    /// Sample or population variance using a numerically stable two-pass centered deviation algorithm backed by Accelerate vDSP.
    ///
    /// ## Numerical Stability
    /// Implements a two-pass centered deviation algorithm:
    /// 1. First pass computes the sample mean $\mu = \frac{1}{N}\sum x_i$ using `vDSP_meanvD`.
    /// 2. Second pass computes centered residuals $x_i - \mu$ via `vDSP_vsaddD` and sums their squares with `vDSP.sumOfSquares`.
    ///
    /// This avoids catastrophic floating-point cancellation inherent to the naive $E[X^2] - (E[X])^2$ formula when
    /// the sample variance is small relative to the squared mean (e.g. $[10^9+1, 10^9+2, 10^9+3]$).
    ///
    /// - Parameters:
    ///   - values: Input numeric sample array.
    ///   - ddof: Delta degrees of freedom (1 = sample variance, 0 = population variance). Defaults to 1.
    ///   - checkNaN: If `true`, throws an error if any input value is NaN or Infinity. Defaults to `true`.
    /// - Throws: ``StatsError/emptyInput`` if array is empty, ``StatsError/invalidDDOF(_:)`` if `ddof < 0`,
    ///   ``StatsError/insufficientData(minimum:got:)`` if `values.count <= ddof`, or ``StatsError/containsNaN`` if `checkNaN` is true and NaNs are present.
    /// - Returns: Sample or population variance.
    public static func variance(_ values: [Double], ddof: Int = 1, checkNaN: Bool = true) throws -> Double {
        try requireNonEmpty(values)
        guard ddof >= 0 else { throw StatsError.invalidDDOF(ddof) }
        let n = Double(values.count)
        guard n > Double(ddof) else { throw StatsError.insufficientData(minimum: ddof + 1, got: values.count) }

        var meanVal = 0.0
        let len = vDSP_Length(values.count)
        vDSP_meanvD(values, 1, &meanVal, len)

        if checkNaN && meanVal.isNaN {
            try requireNoNaN(values)
        }

        var negMean = -meanVal
        var centered = [Double](repeating: 0.0, count: values.count)
        vDSP_vsaddD(values, 1, &negMean, &centered, 1, len)

        let sumSquares = vDSP.sumOfSquares(centered)
        let divisor = n - Double(ddof)
        return Swift.max(0.0, sumSquares / divisor)
    }

    /// Sample or population variance for Float values using a numerically stable two-pass centered deviation algorithm backed by Accelerate vDSP.
    ///
    /// - Parameters:
    ///   - values: Input float sample array.
    ///   - ddof: Delta degrees of freedom (1 = sample variance, 0 = population variance). Defaults to 1.
    ///   - checkNaN: If `true`, throws an error if any input value is NaN or Infinity. Defaults to `true`.
    /// - Throws: ``StatsError/emptyInput``, ``StatsError/invalidDDOF(_:)``, ``StatsError/insufficientData(minimum:got:)``, or ``StatsError/containsNaN``.
    /// - Returns: Sample or population variance.
    public static func variance(_ values: [Float], ddof: Int = 1, checkNaN: Bool = true) throws -> Float {
        guard !values.isEmpty else { throw StatsError.emptyInput }
        guard ddof >= 0 else { throw StatsError.invalidDDOF(ddof) }
        let n = Float(values.count)
        guard n > Float(ddof) else { throw StatsError.insufficientData(minimum: ddof + 1, got: values.count) }

        var meanVal: Float = 0.0
        let len = vDSP_Length(values.count)
        vDSP_meanv(values, 1, &meanVal, len)

        if checkNaN && meanVal.isNaN {
            if values.contains(where: { $0.isNaN }) { throw StatsError.containsNaN }
        }

        var negMean: Float = -meanVal
        var centered = [Float](repeating: 0.0, count: values.count)
        vDSP_vsadd(values, 1, &negMean, &centered, 1, len)

        let sumSquares = vDSP.sumOfSquares(centered)
        let divisor = n - Float(ddof)
        return Swift.max(0.0, sumSquares / divisor)
    }

    /// Standard deviation for Double values.
    /// - Parameters:
    ///   - values: <#description#>
    ///   - ddof: <#description#>
    ///   - checkNaN: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func standardDeviation(_ values: [Double], ddof: Int = 1, checkNaN: Bool = true) throws -> Double {
        try variance(values, ddof: ddof, checkNaN: checkNaN).squareRoot()
    }

    /// Standard deviation for Float values.
    /// - Parameters:
    ///   - values: <#description#>
    ///   - ddof: <#description#>
    ///   - checkNaN: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func standardDeviation(_ values: [Float], ddof: Int = 1, checkNaN: Bool = true) throws -> Float {
        try variance(values, ddof: ddof, checkNaN: checkNaN).squareRoot()
    }

    /// Median via vDSP.sort.
    /// - Parameters:
    ///   - values: <#description#>
    ///   - checkNaN: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func median(_ values: [Double], checkNaN: Bool = true) throws -> Double {
        try requireNonEmpty(values)
        if checkNaN { try requireNoNaN(values) }
        var copy = values
        vDSP.sort(&copy, sortOrder: .ascending)
        let n = copy.count
        if n % 2 == 1 {
            return copy[n / 2]
        } else {
            return (copy[n / 2 - 1] + copy[n / 2]) / 2.0
        }
    }

    /// Float overload for median using vDSP.sort.
    /// - Parameters:
    ///   - values: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func median(_ values: [Float]) throws -> Float {
        guard !values.isEmpty else { throw StatsError.emptyInput }
        var copy = values
        vDSP.sort(&copy, sortOrder: .ascending)
        let n = copy.count
        if n % 2 == 1 {
            return copy[n / 2]
        } else {
            return (copy[n / 2 - 1] + copy[n / 2]) / 2.0
        }
    }

    /// Mode(s) — values with the highest frequency.
    /// - Parameters:
    ///   - values: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func mode(_ values: [Double]) throws -> [Double] {
        try requireNonEmpty(values)
        var freq: [Double: Int] = [:]
        for v in values { freq[v, default: 0] += 1 }
        let maxFreq = freq.values.max() ?? 0
        return freq.filter { $0.value == maxFreq }.keys.sorted()
    }

    /// Percentile using linear interpolation (matches NumPy's default method).
    /// - Parameters:
    ///   - values: <#description#>
    ///   - q: <#description#>
    ///   - checkNaN: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func percentile(_ values: [Double], q: Double, checkNaN: Bool = true) throws -> Double {
        try requireNonEmpty(values)
        if checkNaN { try requireNoNaN(values) }
        guard q >= 0 && q <= 1 else { throw StatsError.invalidPercentile(q) }
        let sorted = values.sorted()
        let n      = Double(sorted.count)
        let idx    = q * (n - 1)
        let lo     = Int(idx)
        let hi     = Swift.min(lo + 1, sorted.count - 1)
        let frac   = idx - Double(lo)
        return sorted[lo] + frac * (sorted[hi] - sorted[lo])
    }

    /// Multiple percentiles at once.
    /// - Parameters:
    ///   - values: <#description#>
    ///   - probs: <#description#>
    ///   - checkNaN: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func quantiles(_ values: [Double], probs: [Double], checkNaN: Bool = true) throws -> [Double] {
        try probs.map { try percentile(values, q: $0, checkNaN: checkNaN) }
    }

    /// Standardised third central moment (Fisher's definition).
    /// - Parameters:
    ///   - values: <#description#>
    ///   - checkNaN: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func skewness(_ values: [Double], checkNaN: Bool = true) throws -> Double {
        try requireNonEmpty(values, minimum: 3)
        if checkNaN { try requireNoNaN(values) }
        let n   = Double(values.count)
        let mu  = vDSP.mean(values)
        let s   = try standardDeviation(values, ddof: 1, checkNaN: checkNaN)
        guard s > 0 else { return 0 }

        let diffs = values.map { $0 - mu }
        let m3  = diffs.map { $0 * $0 * $0 }.reduce(0, +) / n
        return m3 / (s * s * s)
    }

    /// Excess kurtosis (Fisher's definition, normal distribution = 0).
    /// - Parameters:
    ///   - values: <#description#>
    ///   - checkNaN: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func kurtosis(_ values: [Double], checkNaN: Bool = true) throws -> Double {
        try requireNonEmpty(values, minimum: 4)
        if checkNaN { try requireNoNaN(values) }
        let n   = Double(values.count)
        let mu  = vDSP.mean(values)
        let s   = try standardDeviation(values, ddof: 1, checkNaN: checkNaN)
        guard s > 0 else { return 0 }

        let m4 = values.map { pow($0 - mu, 4) }.reduce(0, +) / n
        return m4 / pow(s, 4) - 3.0
    }

    /// Minimum value using vDSP.
    /// - Parameters:
    ///   - values: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func min(_ values: [Double]) throws -> Double {
        try requireNonEmpty(values)
        return vDSP.minimum(values)
    }

    /// Maximum value using vDSP.
    /// - Parameters:
    ///   - values: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func max(_ values: [Double]) throws -> Double {
        try requireNonEmpty(values)
        return vDSP.maximum(values)
    }

    /// Sum using vDSP.
    /// - Parameters:
    ///   - values: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func sum(_ values: [Double]) throws -> Double {
        try requireNonEmpty(values)
        return vDSP.sum(values)
    }

    /// Range (max - min).
    /// - Parameters:
    ///   - values: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func range(_ values: [Double]) throws -> Double {
        try max(values) - min(values)
    }

    /// Comprehensive descriptive statistics summary.
    ///
    /// - Parameters:
    ///   - values: Input numeric sample array.
    ///   - nullCount: Number of missing/null values excluded before passing into this function. Defaults to 0.
    ///   - checkNaN: If `true`, verifies that input array does not contain NaN values before computing metrics. Defaults to `true`.
    /// - Returns: A ``DescriptiveStats`` record containing count, mean, std, variance, min, quartiles, median, max, skewness, and kurtosis.
    /// - Throws: ``StatsError/emptyInput`` or ``StatsError/containsNaN`` if `checkNaN` is `true` and any elements are NaN.
    public static func describe(_ values: [Double], nullCount: Int = 0, checkNaN: Bool = true) throws -> DescriptiveStats {
        try requireNonEmpty(values)
        if checkNaN { try requireNoNaN(values) }
        return DescriptiveStats(
            count:             values.count,
            mean:              try mean(values, checkNaN: checkNaN),
            standardDeviation: try standardDeviation(values, checkNaN: checkNaN),
            variance:          try variance(values, checkNaN: checkNaN),
            min:               try min(values),
            q1:                try percentile(values, q: 0.25, checkNaN: checkNaN),
            median:            try median(values, checkNaN: checkNaN),
            q3:                try percentile(values, q: 0.75, checkNaN: checkNaN),
            max:               try max(values),
            skewness:          try skewness(values, checkNaN: checkNaN),
            kurtosis:          try kurtosis(values, checkNaN: checkNaN),
            nullCount:         nullCount
        )
    }

    // MARK: – Internal validation helpers

    internal static func requireNonEmpty(_ values: [Double], minimum: Int = 1) throws {
        guard values.count >= minimum else {
            if values.isEmpty { throw StatsError.emptyInput }
            throw StatsError.insufficientData(minimum: minimum, got: values.count)
        }
    }

    internal static func requireNoNaN(_ values: [Double]) throws {
        if values.contains(where: { $0.isNaN }) { throw StatsError.containsNaN }
    }

    internal static func requireSameSize(_ a: [Double], _ b: [Double]) throws {
        guard a.count == b.count else {
            throw StatsError.sizeMismatch(sizeA: a.count, sizeB: b.count)
        }
    }
}
