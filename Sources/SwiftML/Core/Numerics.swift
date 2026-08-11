import Foundation

/// Numerically-stable sigmoid, clamped to avoid `exp` overflow on extreme logits.
@inlinable
public func sigmoid(_ x: Double) -> Double {
    1.0 / (1.0 + exp(-Swift.max(-50.0, Swift.min(50.0, x))))
}

extension Array where Element: Comparable {
    /// Index of the maximum element, or 0 for an empty array (matches existing call-site fallback behavior).
    @inlinable
    public func argmax() -> Int {
        enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
    }
}
