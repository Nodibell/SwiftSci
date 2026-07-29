import Foundation
import SwiftDataFrame

/// A seedable pseudo-random number generator for reproducible shuffling.
public struct SeedableRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - seed: The seed.
    public init(seed: Int) {
        self.state = seed == 0 ? 123456789 : UInt64(abs(seed))
    }
    
    /// Next.
    /// - Returns: A `UInt64` result.
    public mutating func next() -> UInt64 {
        state = state.multipliedReportingOverflow(by: 6364136223846793005).partialValue &+ 1442695040888963407
        return state
    }
}

/// Splits features and targets into training and testing subsets.
public func trainTestSplit(
    _ features: [[Double]],
    _ targets: [Double],
    testSize: Double = 0.25,
    shuffle: Bool = true,
    seed: Int? = nil
) throws -> (
    trainFeatures: [[Double]],
    testFeatures: [[Double]],
    trainTargets: [Double],
    testTargets: [Double]
) {
    guard !features.isEmpty else {
        throw PreprocessingError.emptyInput
    }
    guard features.count == targets.count else {
        throw PreprocessingError.dimensionMismatch(expected: features.count, got: targets.count)
    }
    guard testSize > 0.0 && testSize < 1.0 else {
        throw NSError(domain: "TrainTestSplit", code: 1, userInfo: [NSLocalizedDescriptionKey: "testSize must be between 0.0 and 1.0 exclusive."])
    }
    
    let totalCount = features.count
    var indices = Array(0..<totalCount)
    
    if shuffle {
        if let s = seed {
            var rng = SeedableRandomNumberGenerator(seed: s)
            indices.shuffle(using: &rng)
        } else {
            indices.shuffle()
        }
    }
    
    let testCount = Int(Double(totalCount) * testSize)
    let trainCount = totalCount - testCount
    
    guard trainCount > 0, testCount > 0 else {
        throw NSError(domain: "TrainTestSplit", code: 2, userInfo: [NSLocalizedDescriptionKey: "Sample size too small to split with given testSize."])
    }
    
    var trainFeatures = [[Double]]()
    var testFeatures = [[Double]]()
    var trainTargets = [Double]()
    var testTargets = [Double]()
    
    trainFeatures.reserveCapacity(trainCount)
    testFeatures.reserveCapacity(testCount)
    trainTargets.reserveCapacity(trainCount)
    testTargets.reserveCapacity(testCount)
    
    for i in 0..<trainCount {
        let idx = indices[i]
        trainFeatures.append(features[idx])
        trainTargets.append(targets[idx])
    }
    
    for i in trainCount..<totalCount {
        let idx = indices[i]
        testFeatures.append(features[idx])
        testTargets.append(targets[idx])
    }
    
    return (trainFeatures, testFeatures, trainTargets, testTargets)
}

extension DataFrame {
    /// Splits the DataFrame rows into training and testing DataFrames.
    public func trainTestSplit(
        testSize: Double = 0.25,
        shuffle: Bool = true,
        seed: Int? = nil
    ) throws -> (train: DataFrame, test: DataFrame) {
        let nRows = shape.rows
        guard nRows > 0 else {
            return (self, self)
        }
        guard testSize > 0.0 && testSize < 1.0 else {
            throw NSError(domain: "DataFrame.trainTestSplit", code: 1, userInfo: [NSLocalizedDescriptionKey: "testSize must be between 0.0 and 1.0 exclusive."])
        }
        
        var indices = Array(0..<nRows)
        if shuffle {
            if let s = seed {
                var rng = SeedableRandomNumberGenerator(seed: s)
                indices.shuffle(using: &rng)
            } else {
                indices.shuffle()
            }
        }
        
        let testCount = Int(Double(nRows) * testSize)
        let trainCount = nRows - testCount
        
        guard trainCount > 0, testCount > 0 else {
            throw NSError(domain: "DataFrame.trainTestSplit", code: 2, userInfo: [NSLocalizedDescriptionKey: "DataFrame size too small to split."])
        }
        
        let trainIndices = Array(indices[0..<trainCount])
        let testIndices = Array(indices[trainCount..<nRows])
        
        let trainDf = gathered(at: trainIndices)
        let testDf = gathered(at: testIndices)
        
        return (trainDf, testDf)
    }
}
