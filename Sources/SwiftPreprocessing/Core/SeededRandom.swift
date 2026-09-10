import Foundation

/// A high-performance pseudo-random number generator implementing the Xoshiro256++ algorithm.
///
/// ## Statistical Properties
/// Offers a period of 2^256 - 1 and passes BigCrush statistical randomness suites, providing
/// reproducible bootstrapping for ensemble trees, cross-validation shuffling, and stochastic clustering models.
///
/// ## Thread Safety
/// Conforms to `RandomNumberGenerator` and `Sendable` as an immutable value type upon copy,
/// mutating internal state only via mutating methods.
public struct SeededRandom: RandomNumberGenerator, Sendable {
    private var s0: UInt64
    private var s1: UInt64
    private var s2: UInt64
    private var s3: UInt64

    /// Creates a new seeded pseudo-random number generator initialized with the Xoshiro256++ state.
    ///
    /// The 64-bit integer seed is expanded into the four 64-bit state words using SplitMix64.
    ///
    /// - Parameter seed: The integer seed value.
    ///
    /// ## Complexity
    /// O(1) constant time initialization.
    ///
    /// ## Thread Safety
    /// Independent instances are completely isolated and safe to transfer across concurrency domains.
    public init(seed: Int) {
        var smState = UInt64(bitPattern: Int64(seed &+ 1))
        
        @inline(__always)
        func splitmix64(state: inout UInt64) -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        
        let a = splitmix64(state: &smState)
        let b = splitmix64(state: &smState)
        let c = splitmix64(state: &smState)
        let d = splitmix64(state: &smState)
        
        if (a | b | c | d) == 0 {
            self.s0 = 1
            self.s1 = 0
            self.s2 = 0
            self.s3 = 0
        } else {
            self.s0 = a
            self.s1 = b
            self.s2 = c
            self.s3 = d
        }
    }

    /// Advances the generator state and returns the next pseudo-random 64-bit unsigned integer.
    ///
    /// Conforms to `RandomNumberGenerator.next()`.
    ///
    /// - Returns: A uniformly distributed `UInt64` value.
    ///
    /// ## Complexity
    /// O(1) consisting of bit shifts, rotations, and XOR operations without branching or division.
    @inline(__always)
    public mutating func next() -> UInt64 {
        let result = rotl(s0 &+ s3, 23) &+ s0
        let t = s1 << 17
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = rotl(s3, 45)
        return result
    }

    /// Returns a pseudo-random integer in the half-open range `0..<upperBound`.
    ///
    /// - Parameter upperBound: The exclusive upper bound, must be greater than zero.
    /// - Returns: An integer value in `0..<upperBound`. Returns 0 if `upperBound <= 0`.
    ///
    /// ## Complexity
    /// O(1) constant time.
    public mutating func nextInt(upperBound: Int) -> Int {
        guard upperBound > 0 else { return 0 }
        return Int(next() % UInt64(upperBound))
    }

    /// Returns a uniformly distributed pseudo-random `Double` in the half-open interval `[0, 1)`.
    ///
    /// - Returns: A IEEE 754 floating-point value in `0.0..<1.0`.
    ///
    /// ## Complexity
    /// O(1) constant time.
    public mutating func nextDouble() -> Double {
        Double(next() >> 11) * (1.0 / Double(UInt64(1) << 53))
    }

    @inline(__always)
    private func rotl(_ x: UInt64, _ k: UInt64) -> UInt64 {
        (x << k) | (x >> (64 - k))
    }
}
