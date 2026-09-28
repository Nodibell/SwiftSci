import Accelerate
import SwiftDataFrame

// MARK: – Linear Algebra (vDSP + LAPACK)

extension Stats {

    /// Returns a dot product using the performance policy.
    ///
    /// The reduction order is implementation-dependent. Use the explicit accuracy
    /// overload when cancellation requires compensated accumulation.
    /// - Throws: `StatsError.emptyInput` or a size-mismatch error.
    public static func dotProduct(_ a: [Double], _ b: [Double]) throws -> Double {
        try dotProduct(a, b, accuracy: .performance)
    }

    /// Returns a dot product with the selected arithmetic policy.
    ///
    /// Both policies accept and return `Double`. NaN propagates; opposite infinite
    /// contributions and zero multiplied by infinity produce NaN. Compensated mode
    /// retains rounding corrections and uses exact products for unsafe ranges or
    /// severe cancellation. It does not promise universally correct rounding.
    /// - Parameters:
    ///   - a: First vector.
    ///   - b: Second vector, with the same number of elements as `a`.
    ///   - accuracy: Throughput or compensated accumulation.
    /// - Throws: `StatsError.emptyInput` or a size-mismatch error.
    public static func dotProduct(
        _ a: [Double], _ b: [Double], accuracy: DotProductAccuracy
    ) throws -> Double {
        try requireNonEmpty(a)
        try requireSameSize(a, b)
        switch accuracy {
        case .performance:
            // Small vectors favor vDSP. Avoid narrowing counts outside CBLAS's Int32 range.
            if a.count >= 4096 && a.count <= Int(Int32.max) {
                return cblas_ddot(Int32(a.count), a, 1, b, 1)
            }
            return vDSP.dot(a, b)
        case .compensated:
            return CompensatedDotProduct.evaluate(a, b)
        }
    }

    /// Vector norm: L1, L2 (Euclidean), or L∞.
    /// - Parameters:
    ///   - values: Numeric values array to be evaluated.
    ///   - order: Mathematical norm order (.l1, .l2, or .infinity) or ARIMA order tuple.
    /// - Throws: `StatsError` if input is empty, contains NaNs when forbidden, or dimensions mismatch.
    /// - Returns: Calculated mathematical vector or matrix norm.
    public static func norm(_ values: [Double], order: NormOrder = .l2) throws -> Double {
        try requireNonEmpty(values)
        switch order {
        case .l1:
            return cblas_dasum(Int32(values.count), values, 1)
        case .l2:
            return vDSP.sumOfSquares(values).squareRoot()
        case .infinity:
            return values.map(abs).max() ?? 0
        }
    }

    /// Cosine similarity: dot(a,b) / (‖a‖ · ‖b‖).
    /// - Parameters:
    ///   - a: First vector or numeric operand.
    ///   - b: Second vector or numeric operand.
    /// - Throws: `StatsError` if input is empty, contains NaNs when forbidden, or dimensions mismatch.
    /// - Returns: Normalized similarity metric between -1.0 and 1.0.
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
    /// - Parameters:
    ///   - a: First vector or numeric operand.
    ///   - b: Second vector or numeric operand.
    /// - Throws: `StatsError` if input is empty, contains NaNs when forbidden, or dimensions mismatch.
    /// - Returns: Array of computed numeric values.
    public static func add(_ a: [Double], _ b: [Double]) throws -> [Double] {
        try requireSameSize(a, b)
        var result = [Double](repeating: 0, count: a.count)
        vDSP.add(a, b, result: &result)
        return result
    }

    /// Element-wise subtract.
    /// - Parameters:
    ///   - a: First vector or numeric operand.
    ///   - b: Second vector or numeric operand.
    /// - Throws: `StatsError` if input is empty, contains NaNs when forbidden, or dimensions mismatch.
    /// - Returns: Array of computed numeric values.
    public static func subtract(_ a: [Double], _ b: [Double]) throws -> [Double] {
        try requireSameSize(a, b)
        var result = [Double](repeating: 0, count: a.count)
        vDSP.subtract(b, a, result: &result)
        return result
    }

    /// Scalar multiplication.
    /// - Parameters:
    ///   - values: Numeric values array to be evaluated.
    ///   - scalar: Scalar multiplier or shift value.
    /// - Returns: Array of computed numeric values.
    public static func scale(_ values: [Double], by scalar: Double) -> [Double] {
        vDSP.multiply(scalar, values)
    }

    /// Normalise a vector to unit L2 norm.
    /// - Parameters:
    ///   - values: Numeric values array to be evaluated.
    /// - Throws: `StatsError` if input is empty, contains NaNs when forbidden, or dimensions mismatch.
    /// - Returns: Array of computed numeric values.
    public static func normalise(_ values: [Double]) throws -> [Double] {
        let n = try norm(values, order: .l2)
        guard n > 0 else { throw StatsError.divisionByZero(context: "normalise") }
        return vDSP.multiply(1.0 / n, values)
    }

    /// Standardise values: (x - mean) / std.
    /// - Parameters:
    ///   - values: Numeric values array to be evaluated.
    /// - Throws: `StatsError` if input is empty, contains NaNs when forbidden, or dimensions mismatch.
    /// - Returns: Array of computed numeric values.
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
