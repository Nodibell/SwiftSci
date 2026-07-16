import Accelerate

// MARK: – Linear Algebra (vDSP + LAPACK)

extension Stats {

    /// Dot product of two vectors using vDSP.dot.
    public static func dotProduct(_ a: [Double], _ b: [Double]) throws -> Double {
        try requireNonEmpty(a)
        try requireSameSize(a, b)
        return vDSP.dot(a, b)
    }

    /// Vector norm: L1, L2 (Euclidean), or L∞.
    public static func norm(_ values: [Double], order: NormOrder = .l2) throws -> Double {
        try requireNonEmpty(values)
        switch order {
        case .l1:
            return values.reduce(0) { $0 + abs($1) }
        case .l2:
            return vDSP.sumOfSquares(values).squareRoot()
        case .infinity:
            return values.map(abs).max() ?? 0
        }
    }

    /// Cosine similarity: dot(a,b) / (‖a‖ · ‖b‖).
    public static func cosineSimilarity(_ a: [Double], _ b: [Double]) throws -> Double {
        try requireNonEmpty(a)
        try requireSameSize(a, b)
        let dot  = vDSP.dot(a, b)
        let normA = vDSP.sumOfSquares(a).squareRoot()
        let normB = vDSP.sumOfSquares(b).squareRoot()
        guard normA > 0 && normB > 0 else {
            throw StatsError.divisionByZero(context: "cosineSimilarity")
        }
        return Swift.max(-1.0, Swift.min(1.0, dot / (normA * normB)))
    }

    /// Element-wise add.
    public static func add(_ a: [Double], _ b: [Double]) throws -> [Double] {
        try requireSameSize(a, b)
        var result = [Double](repeating: 0, count: a.count)
        vDSP.add(a, b, result: &result)
        return result
    }

    /// Element-wise subtract.
    public static func subtract(_ a: [Double], _ b: [Double]) throws -> [Double] {
        try requireSameSize(a, b)
        var result = [Double](repeating: 0, count: a.count)
        vDSP.subtract(b, a, result: &result)
        return result
    }

    /// Scalar multiplication.
    public static func scale(_ values: [Double], by scalar: Double) -> [Double] {
        vDSP.multiply(scalar, values)
    }

    /// Normalise a vector to unit L2 norm.
    public static func normalise(_ values: [Double]) throws -> [Double] {
        let n = try norm(values, order: .l2)
        guard n > 0 else { throw StatsError.divisionByZero(context: "normalise") }
        return vDSP.multiply(1.0 / n, values)
    }

    /// Standardise values: (x - mean) / std.
    public static func standardise(_ values: [Double]) throws -> [Double] {
        try requireNonEmpty(values, minimum: 2)
        let mu = vDSP.mean(values)
        let s  = try standardDeviation(values, ddof: 1)
        guard s > 0 else { throw StatsError.divisionByZero(context: "standardise") }
        var centred = [Double](repeating: 0, count: values.count)
        vDSP.add(-mu, values, result: &centred)
        return vDSP.multiply(1.0 / s, centred)
    }
}
