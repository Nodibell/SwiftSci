import Foundation

/// Self-contained deterministic pseudo-random number generator for tests.
struct SimpleRNG {
    private var state: UInt64
    
    init(seed: UInt64) {
        self.state = seed
    }
    
    mutating func nextDouble() -> Double {
        state = state &* 6364136223846793005 &+ 1
        return Double(state) / Double(UInt64.max)
    }
    
    mutating func nextGaussian() -> Double {
        let u1 = nextDouble()
        let u2 = nextDouble()
        let r = (-2.0 * log(u1 > 0 ? u1 : 1e-15)).squareRoot()
        let theta = 2.0 * Double.pi * u2
        return r * cos(theta)
    }
}
