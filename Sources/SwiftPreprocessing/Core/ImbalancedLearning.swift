import Foundation

/// Container for resampled dataset features and targets.
public struct ResampledDataset: Sendable {
    /// The features.
    public let features: [[Double]]
    /// The targets.
    public let targets: [Double]
    /// Creates a new instance.
    /// - Parameters:
    ///   - features: The features.
    ///   - targets: The targets.
    public init(features: [[Double]], targets: [Double]) {
        self.features = features
        self.targets = targets
    }
}

/// Synthetic Minority Over-sampling Technique (SMOTE) for imbalanced classification datasets.
public final class SMOTE: Sendable {
    /// The k neighbors.
    public let kNeighbors: Int
    /// The seed.
    public let seed: UInt64
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - kNeighbors: The k neighbors.
    ///   - seed: The seed.
    public init(kNeighbors: Int = 5, seed: UInt64 = 42) {
        self.kNeighbors = max(1, kNeighbors)
        self.seed = seed
    }
    
    /// Resamples minority class instances by generating synthetic samples along line segments connecting k-nearest neighbors.
    public func fitResample(features: [[Double]], targets: [Double]) throws -> ResampledDataset {
        guard !features.isEmpty, features.count == targets.count else {
            throw PreprocessingError.emptyInput
        }
        
        let numFeatures = features[0].count
        var classCounts: [Double: Int] = [:]
        var classIndices: [Double: [Int]] = [:]
        
        for (i, t) in targets.enumerated() {
            classCounts[t, default: 0] += 1
            classIndices[t, default: []].append(i)
        }
        
        guard let maxCount = classCounts.values.max() else {
            return ResampledDataset(features: features, targets: targets)
        }
        
        var syntheticFeatures = features
        var syntheticTargets = targets
        var rng = SeededRandom(seed: Int(self.seed))
        
        for (cls, indices) in classIndices {
            let count = indices.count
            guard count < maxCount else { continue }
            let samplesToGenerate = maxCount - count
            guard count > 1 else { continue }
            let k = min(kNeighbors, count - 1)
            
            let classFeatures = indices.map { features[$0] }
            
            for _ in 0..<samplesToGenerate {
                let idx = rng.nextInt(upperBound: count)
                let baseSample = classFeatures[idx]
                
                // Find k-nearest neighbors within minority class
                var distances: [(index: Int, dist: Double)] = []
                for (otherIdx, otherSample) in classFeatures.enumerated() where otherIdx != idx {
                    var d = 0.0
                    for f in 0..<numFeatures {
                        let diff = baseSample[f] - otherSample[f]
                        d += diff * diff
                    }
                    distances.append((otherIdx, d))
                }
                
                distances.sort { $0.dist < $1.dist }
                let neighborIdx = distances[rng.nextInt(upperBound: k)].index
                let neighborSample = classFeatures[neighborIdx]
                
                // Interpolate along line segment using seeded random gap
                let gap = rng.nextDouble()
                var newSample = [Double](repeating: 0.0, count: numFeatures)
                for f in 0..<numFeatures {
                    newSample[f] = baseSample[f] + gap * (neighborSample[f] - baseSample[f])
                }
                
                syntheticFeatures.append(newSample)
                syntheticTargets.append(cls)
            }
        }
        
        return ResampledDataset(features: syntheticFeatures, targets: syntheticTargets)
    }
}

/// Random Undersampler for balancing class distribution by random sub-sampling of majority classes.
public final class RandomUndersampler: Sendable {
    /// The seed.
    public let seed: UInt64
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - seed: The seed.
    public init(seed: UInt64 = 42) {
        self.seed = seed
    }
    
    /// Fit resample.
    /// - Parameters:
    ///   - features: The features.
    ///   - targets: The targets.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `ResampledDataset` result.
    public func fitResample(features: [[Double]], targets: [Double]) throws -> ResampledDataset {
        guard !features.isEmpty, features.count == targets.count else {
            throw PreprocessingError.emptyInput
        }
        
        var classIndices: [Double: [Int]] = [:]
        for (i, t) in targets.enumerated() {
            classIndices[t, default: []].append(i)
        }
        
        guard let minCount = classIndices.values.map({ $0.count }).min() else {
            return ResampledDataset(features: features, targets: targets)
        }
        
        var rng = SeededRandom(seed: Int(self.seed))
        var resampledIndices: [Int] = []
        
        for (_, indices) in classIndices {
            // Seeded Fisher-Yates shuffle
            var shuffled = indices
            for i in stride(from: shuffled.count - 1, through: 1, by: -1) {
                let j = rng.nextInt(upperBound: i + 1)
                shuffled.swapAt(i, j)
            }
            resampledIndices.append(contentsOf: shuffled.prefix(minCount))
        }
        
        resampledIndices.sort()
        let resampledFeatures = resampledIndices.map { features[$0] }
        let resampledTargets = resampledIndices.map { targets[$0] }
        
        return ResampledDataset(features: resampledFeatures, targets: resampledTargets)
    }
}
