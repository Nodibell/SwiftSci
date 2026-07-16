import Foundation

/// Simple seeded LCG random number generator (Sendable, value type).
/// Used for reproducible bootstrapping and shuffling in SwiftML and SwiftOptimize.
public struct SeededRandom: Sendable {
    private var state: UInt64

    public init(seed: Int) {
        self.state = UInt64(bitPattern: Int64(seed &+ 1))
    }

    /// Returns a random integer in `0..<upperBound`.
    public mutating func nextInt(upperBound: Int) -> Int {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Int(state >> 33) % upperBound
    }
}
