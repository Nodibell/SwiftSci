import Foundation

/// Simple seeded LCG random number generator (Sendable, value type).
/// Used for reproducible bootstrapping, shuffling, and oversampling.
public struct SeededRandom: Sendable {
    private var state: UInt64

    /// Creates a new instance.
    /// - Parameters:
    ///   - seed: The seed.
    public init(seed: Int) {
        self.state = UInt64(bitPattern: Int64(seed &+ 1))
    }

    /// Advances the LCG state and returns the raw 64-bit value.
    public mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }

    /// Returns a random integer in `0..<upperBound`.
    public mutating func nextInt(upperBound: Int) -> Int {
        guard upperBound > 0 else { return 0 }
        return Int(next() >> 33) % upperBound
    }

    /// Returns a random Double in [0, 1).
    public mutating func nextDouble() -> Double {
        Double(next() >> 11) / Double(1 << 53)
    }
}
