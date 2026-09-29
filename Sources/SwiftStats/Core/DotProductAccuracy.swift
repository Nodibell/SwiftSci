/// Selects dot-product arithmetic without changing the input or output type.
public enum DotProductAccuracy: Sendable {
    /// Prioritizes throughput using Accelerate. Reduction order and final rounding
    /// bits can differ across input sizes, processors, or platform versions.
    case performance

    /// Retains product and addition rounding corrections. Unsafe exponent ranges
    /// and severe cancellation use an exact integer accumulator with one final
    /// rounding. Ordinary cases use SIMD compensation, so this policy does not
    /// guarantee universally correct rounding or cross-platform bitwise identity.
    case compensated
}
