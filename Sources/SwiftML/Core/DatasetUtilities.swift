import Foundation

/// Synthetic dataset generation utilities for testing and benchmarking machine learning estimators.
public enum DatasetUtilities {
    /// Generates a synthetic classification dataset with separable gaussian clusters.
    public static func makeClassification(
        nSamples: Int = 100,
        nFeatures: Int = 2,
        nClasses: Int = 2,
        seed: UInt64 = 42
    ) -> (features: [[Double]], targets: [Double]) {
        var rng = SeededRandom(seed: Int(seed))
        var features: [[Double]] = []
        var targets: [Double] = []
        
        let samplesPerClass = nSamples / nClasses
        
        for c in 0..<nClasses {
            let center = (0..<nFeatures).map { _ in Double(c * 3) }
            for _ in 0..<samplesPerClass {
                let row = center.map { $0 + (Double(rng.nextInt(upperBound: 2000)) / 1000.0 - 1.0) }
                features.append(row)
                targets.append(Double(c))
            }
        }
        
        return (features: features, targets: targets)
    }
    
    /// Generates a synthetic linear regression dataset $y = X \cdot w + b + \epsilon$.
    public static func makeRegression(
        nSamples: Int = 100,
        nFeatures: Int = 2,
        noise: Double = 0.1,
        seed: UInt64 = 42
    ) -> (features: [[Double]], targets: [Double]) {
        var rng = SeededRandom(seed: Int(seed))
        let weights = (0..<nFeatures).map { Double($0 + 1) * 1.5 }
        let bias = 2.5
        
        var features: [[Double]] = []
        var targets: [Double] = []
        
        for _ in 0..<nSamples {
            let row = (0..<nFeatures).map { _ in Double(rng.nextInt(upperBound: 4000)) / 1000.0 - 2.0 }
            var y = bias
            for f in 0..<nFeatures {
                y += row[f] * weights[f]
            }
            let n = (Double(rng.nextInt(upperBound: 2000)) / 1000.0 - 1.0) * noise
            y += n
            features.append(row)
            targets.append(y)
        }
        
        return (features: features, targets: targets)
    }
    
    /// Generates a synthetic non-linear 2D dataset of two interleaving half moons.
    public static func makeMoons(
        nSamples: Int = 100,
        noise: Double = 0.1,
        seed: UInt64 = 42
    ) -> (features: [[Double]], targets: [Double]) {
        var rng = SeededRandom(seed: Int(seed))
        let nHalf = nSamples / 2
        var features: [[Double]] = []
        var targets: [Double] = []
        
        for i in 0..<nHalf {
            let angle = Double.pi * Double(i) / Double(nHalf)
            let n1 = (Double(rng.nextInt(upperBound: 2000)) / 1000.0 - 1.0) * noise
            let n2 = (Double(rng.nextInt(upperBound: 2000)) / 1000.0 - 1.0) * noise
            let x = cos(angle) + n1
            let y = sin(angle) + n2
            features.append([x, y])
            targets.append(0.0)
        }
        
        for i in 0..<nHalf {
            let angle = Double.pi * Double(i) / Double(nHalf)
            let n1 = (Double(rng.nextInt(upperBound: 2000)) / 1000.0 - 1.0) * noise
            let n2 = (Double(rng.nextInt(upperBound: 2000)) / 1000.0 - 1.0) * noise
            let x = 1.0 - cos(angle) + n1
            let y = 0.5 - sin(angle) + n2
            features.append([x, y])
            targets.append(1.0)
        }
        
        return (features: features, targets: targets)
    }
}
