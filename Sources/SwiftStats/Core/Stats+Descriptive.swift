import Accelerate

// MARK: – Descriptive Statistics (vDSP-backed)

extension Stats {

    /// Arithmetic mean using vDSP.mean.
    public static func mean(_ values: [Double]) throws -> Double {
        try requireNonEmpty(values)
        try requireNoNaN(values)
        return vDSP.mean(values)
    }

    /// Float overload.
    public static func mean(_ values: [Float]) throws -> Float {
        guard !values.isEmpty else { throw StatsError.emptyInput }
        return vDSP.mean(values)
    }

    /// Sample or population variance.
    /// - Parameter ddof: Delta degrees of freedom (1 = sample, 0 = population).
    public static func variance(_ values: [Double], ddof: Int = 1) throws -> Double {
        try requireNonEmpty(values)
        try requireNoNaN(values)
        guard ddof >= 0 else { throw StatsError.invalidDDOF(ddof) }
        let n = Double(values.count)
        guard n > Double(ddof) else { throw StatsError.insufficientData(minimum: ddof + 1, got: values.count) }

        let mu  = vDSP.mean(values)
        // Σ(x - μ)²
        var centered = [Double](repeating: 0, count: values.count)
        vDSP.add(-mu, values, result: &centered)
        let ssq = vDSP.sumOfSquares(centered)
        let v = ssq / (n - Double(ddof))
        guard v >= 0 else { throw StatsError.negativeVariance }
        return v
    }

    /// Standard deviation.
    public static func standardDeviation(_ values: [Double], ddof: Int = 1) throws -> Double {
        try variance(values, ddof: ddof).squareRoot()
    }

    /// Median via sorted linear interpolation.
    public static func median(_ values: [Double]) throws -> Double {
        try requireNonEmpty(values)
        try requireNoNaN(values)
        let sorted = values.sorted()
        let n = sorted.count
        if n % 2 == 1 {
            return sorted[n / 2]
        } else {
            return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        }
    }

    /// Mode(s) — values with the highest frequency.
    public static func mode(_ values: [Double]) throws -> [Double] {
        try requireNonEmpty(values)
        var freq: [Double: Int] = [:]
        for v in values { freq[v, default: 0] += 1 }
        let maxFreq = freq.values.max() ?? 0
        return freq.filter { $0.value == maxFreq }.keys.sorted()
    }

    /// Percentile using linear interpolation (matches NumPy's default method).
    public static func percentile(_ values: [Double], q: Double) throws -> Double {
        try requireNonEmpty(values)
        try requireNoNaN(values)
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
    public static func quantiles(_ values: [Double], probs: [Double]) throws -> [Double] {
        try probs.map { try percentile(values, q: $0) }
    }

    /// Standardised third central moment (Fisher's definition).
    public static func skewness(_ values: [Double]) throws -> Double {
        try requireNonEmpty(values, minimum: 3)
        try requireNoNaN(values)
        let n   = Double(values.count)
        let mu  = vDSP.mean(values)
        let s   = try standardDeviation(values, ddof: 1)
        guard s > 0 else { return 0 }

        let diffs = values.map { $0 - mu }
        let m3  = diffs.map { $0 * $0 * $0 }.reduce(0, +) / n
        return m3 / (s * s * s)
    }

    /// Excess kurtosis (Fisher's definition, normal distribution = 0).
    public static func kurtosis(_ values: [Double]) throws -> Double {
        try requireNonEmpty(values, minimum: 4)
        try requireNoNaN(values)
        let n   = Double(values.count)
        let mu  = vDSP.mean(values)
        let s   = try standardDeviation(values, ddof: 1)
        guard s > 0 else { return 0 }

        let m4 = values.map { pow($0 - mu, 4) }.reduce(0, +) / n
        return m4 / pow(s, 4) - 3.0
    }

    /// Minimum value using vDSP.
    public static func min(_ values: [Double]) throws -> Double {
        try requireNonEmpty(values)
        return vDSP.minimum(values)
    }

    /// Maximum value using vDSP.
    public static func max(_ values: [Double]) throws -> Double {
        try requireNonEmpty(values)
        return vDSP.maximum(values)
    }

    /// Sum using vDSP.
    public static func sum(_ values: [Double]) throws -> Double {
        try requireNonEmpty(values)
        return vDSP.sum(values)
    }

    /// Range (max - min).
    public static func range(_ values: [Double]) throws -> Double {
        try max(values) - min(values)
    }

    /// Comprehensive descriptive statistics summary.
    public static func describe(_ values: [Double], nullCount: Int = 0) throws -> DescriptiveStats {
        try requireNonEmpty(values)
        try requireNoNaN(values)
        return DescriptiveStats(
            count:             values.count,
            mean:              try mean(values),
            standardDeviation: try standardDeviation(values),
            variance:          try variance(values),
            min:               try min(values),
            q1:                try percentile(values, q: 0.25),
            median:            try median(values),
            q3:                try percentile(values, q: 0.75),
            max:               try max(values),
            skewness:          try skewness(values),
            kurtosis:          try kurtosis(values),
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
