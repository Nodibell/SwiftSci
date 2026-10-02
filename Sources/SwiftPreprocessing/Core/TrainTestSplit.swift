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

/// Splits features and targets into training and testing subsets, optionally preserving class proportions.
/// - Parameters:
///   - features: 2D array of input feature vectors of shape `[N, P]`.
///   - targets: 1D array of ground-truth target values of length `N`.
///   - testSize: Proportion or absolute count of dataset allocated to test split.
///   - stratify: Optional array of class labels for stratified sampling. If provided, class ratios are preserved in train and test splits.
///   - shuffle: Whether to shuffle observations prior to splitting or processing.
///   - seed: Random number generator seed for deterministic reproducibility.
/// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
/// - Returns: The computed ( trainFeatures: [[Double]], testFeatures: [[Double]], trainTargets: [Double], testTargets: [Double] ) result instance.
public func trainTestSplit(
    _ features: [[Double]],
    _ targets: [Double],
    testSize: Double = 0.25,
    stratify: [Double]? = nil,
    shuffle: Bool = true,
    seed: Int? = nil
) throws -> (
    trainFeatures: [[Double]],
    testFeatures: [[Double]],
    trainTargets: [Double],
    testTargets: [Double]
) {
    guard testSize > 0.0 && testSize < 1.0 else {
        throw PreprocessingError.invalidParameter("testSize must be between 0.0 and 1.0 exclusive, got \(testSize).")
    }
    guard !features.isEmpty else {
        throw PreprocessingError.emptyInput
    }
    guard features.count == targets.count else {
        throw PreprocessingError.dimensionMismatch(expected: features.count, got: targets.count)
    }
    if let strat = stratify {
        guard strat.count == features.count else {
            throw PreprocessingError.dimensionMismatch(expected: features.count, got: strat.count)
        }
    }
    
    let totalCount = features.count
    if totalCount == 1 {
        // When dataset has only a single observation, splitting is strictly all-or-nothing:
        // allocated to test if testSize >= 0.5, else allocated to train.
        if testSize >= 0.5 {
            return (trainFeatures: [], testFeatures: features, trainTargets: [], testTargets: targets)
        } else {
            return (trainFeatures: features, testFeatures: [], trainTargets: targets, testTargets: [])
        }
    }
    
    var trainIndices = [Int]()
    var testIndices = [Int]()
    
    var rng: SeedableRandomNumberGenerator? = seed != nil ? SeedableRandomNumberGenerator(seed: seed!) : nil
    
    if let strat = stratify {
        var classIndices: [Double: [Int]] = [:]
        for (i, c) in strat.enumerated() {
            classIndices[c, default: []].append(i)
        }
        
        for (_, indices) in classIndices {
            var clsIdxs = indices
            if shuffle {
                if rng != nil {
                    clsIdxs.shuffle(using: &rng!)
                } else {
                    clsIdxs.shuffle()
                }
            }
            if clsIdxs.count == 1 {
                // G-022: Small-sample resilience for single-observation classes
                if testSize >= 0.5 {
                    testIndices.append(contentsOf: clsIdxs)
                } else {
                    trainIndices.append(contentsOf: clsIdxs)
                }
            } else {
                let clsTestCount = Swift.max(1, Swift.min(clsIdxs.count - 1, Int(round(Double(clsIdxs.count) * testSize))))
                let clsTrainCount = clsIdxs.count - clsTestCount
                trainIndices.append(contentsOf: clsIdxs.prefix(clsTrainCount))
                testIndices.append(contentsOf: clsIdxs.suffix(clsTestCount))
            }
        }
        
        // G-022: Guarantee both train and test have at least one sample when totalCount >= 2
        if trainIndices.isEmpty && !testIndices.isEmpty {
            trainIndices.append(testIndices.removeLast())
        } else if testIndices.isEmpty && !trainIndices.isEmpty {
            testIndices.append(trainIndices.removeLast())
        }
        
        if shuffle {
            if rng != nil {
                trainIndices.shuffle(using: &rng!)
                testIndices.shuffle(using: &rng!)
            } else {
                trainIndices.shuffle()
                testIndices.shuffle()
            }
        }
    } else {
        var indices = Array(0..<totalCount)
        if shuffle {
            if rng != nil {
                indices.shuffle(using: &rng!)
            } else {
                indices.shuffle()
            }
        }
        
        // G-022: Guard against testCount == 0 on small sample sizes (N < 8)
        let testCount = Swift.max(1, Swift.min(totalCount - 1, Int(round(Double(totalCount) * testSize))))
        let trainCount = totalCount - testCount
        
        trainIndices = Array(indices[0..<trainCount])
        testIndices = Array(indices[trainCount..<totalCount])
    }
    
    var trainFeatures = [[Double]]()
    var testFeatures = [[Double]]()
    var trainTargets = [Double]()
    var testTargets = [Double]()
    
    trainFeatures.reserveCapacity(trainIndices.count)
    testFeatures.reserveCapacity(testIndices.count)
    trainTargets.reserveCapacity(trainIndices.count)
    testTargets.reserveCapacity(testIndices.count)
    
    for idx in trainIndices {
        trainFeatures.append(features[idx])
        trainTargets.append(targets[idx])
    }
    
    for idx in testIndices {
        testFeatures.append(features[idx])
        testTargets.append(targets[idx])
    }
    
    return (trainFeatures, testFeatures, trainTargets, testTargets)
}

extension DataFrame {
    /// Splits the DataFrame rows into training and testing DataFrames, optionally stratified by a column.
    /// - Parameters:
    ///   - testSize: Proportion or absolute count of dataset allocated to test split.
    ///   - stratifyColumn: Optional column name to preserve class distribution across train and test splits.
    ///   - shuffle: Whether to shuffle observations prior to splitting or processing.
    ///   - seed: Random number generator seed for deterministic reproducibility.
    /// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
    /// - Returns: A tuple of `(train: DataFrame, test: DataFrame)`.
    public func trainTestSplit(
        testSize: Double = 0.25,
        stratifyColumn: String? = nil,
        shuffle: Bool = true,
        seed: Int? = nil
    ) throws -> (train: DataFrame, test: DataFrame) {
        guard testSize > 0.0 && testSize < 1.0 else {
            throw PreprocessingError.invalidParameter("testSize must be between 0.0 and 1.0 exclusive, got \(testSize).")
        }
        let nRows = shape.rows
        guard nRows > 0 else {
            throw PreprocessingError.emptyInput
        }
        if nRows == 1 {
            // When dataset has only a single observation, splitting is strictly all-or-nothing:
            // allocated to test if testSize >= 0.5, else allocated to train.
            if testSize >= 0.5 {
                return (train: gathered(at: []), test: self)
            } else {
                return (train: self, test: gathered(at: []))
            }
        }
        
        var trainIndices = [Int]()
        var testIndices = [Int]()
        var rng: SeedableRandomNumberGenerator? = seed != nil ? SeedableRandomNumberGenerator(seed: seed!) : nil
        
        if let stratCol = stratifyColumn {
            guard let col = self[stratCol] else {
                throw PreprocessingError.columnNotFound(stratCol)
            }
            
            let strVals = col.toStrings()
            var classIndices: [String: [Int]] = [:]
            for i in 0..<min(nRows, strVals.count) {
                let key = strVals[i]
                classIndices[key, default: []].append(i)
            }
            
            for (_, indices) in classIndices {
                var clsIdxs = indices
                if shuffle {
                    if rng != nil {
                        clsIdxs.shuffle(using: &rng!)
                    } else {
                        clsIdxs.shuffle()
                    }
                }
                if clsIdxs.count == 1 {
                    // G-022: Small-sample resilience for single-observation classes
                    if testSize >= 0.5 {
                        testIndices.append(contentsOf: clsIdxs)
                    } else {
                        trainIndices.append(contentsOf: clsIdxs)
                    }
                } else {
                    let clsTestCount = Swift.max(1, Swift.min(clsIdxs.count - 1, Int(round(Double(clsIdxs.count) * testSize))))
                    let clsTrainCount = clsIdxs.count - clsTestCount
                    trainIndices.append(contentsOf: clsIdxs.prefix(clsTrainCount))
                    testIndices.append(contentsOf: clsIdxs.suffix(clsTestCount))
                }
            }
            
            // G-022: Guarantee both train and test have at least one sample when nRows >= 2
            if trainIndices.isEmpty && !testIndices.isEmpty {
                trainIndices.append(testIndices.removeLast())
            } else if testIndices.isEmpty && !trainIndices.isEmpty {
                testIndices.append(trainIndices.removeLast())
            }
            
            if shuffle {
                if rng != nil {
                    trainIndices.shuffle(using: &rng!)
                    testIndices.shuffle(using: &rng!)
                } else {
                    trainIndices.shuffle()
                    testIndices.shuffle()
                }
            }
        } else {
            var indices = Array(0..<nRows)
            if shuffle {
                if rng != nil {
                    indices.shuffle(using: &rng!)
                } else {
                    indices.shuffle()
                }
            }
            
            // G-022: Guard against testCount == 0 on small sample sizes (N < 8)
            let testCount = Swift.max(1, Swift.min(nRows - 1, Int(round(Double(nRows) * testSize))))
            let trainCount = nRows - testCount
            
            trainIndices = Array(indices[0..<trainCount])
            testIndices = Array(indices[trainCount..<nRows])
        }
        
        let trainDf = gathered(at: trainIndices)
        let testDf = gathered(at: testIndices)
        
        return (trainDf, testDf)
    }
}
