import Foundation

/// Stratified K-Folds cross-validator.
///
/// Provides train/test indices to split data into train/test sets, preserving the percentage of samples for each class.
public struct StratifiedKFold: Sendable {
    /// Number of folds. Must be at least 2.
    public let nSplits: Int
    /// Whether to shuffle each class's samples before splitting into batches.
    public let shuffle: Bool
    /// When shuffle is true, seed determines the generator's state for reproducible results.
    public let seed: Int?
    
    /// Initializes a new StratifiedKFold cross-validator.
    /// - Parameters:
    ///   - nSplits: Number of folds. Defaults to 5.
    ///   - shuffle: Whether to shuffle data within each stratum before splitting. Defaults to true.
    ///   - seed: Random seed for deterministic reproducibility.
    public init(nSplits: Int = 5, shuffle: Bool = true, seed: Int? = nil) {
        self.nSplits = max(2, nSplits)
        self.shuffle = shuffle
        self.seed = seed
    }
    
    /// Generates indices to split data into training and test set folds.
    /// - Parameter targets: Ground-truth target array of length N.
    /// - Throws: `PreprocessingError` if targets are empty or smaller than nSplits.
    /// - Returns: An array of tuples containing `trainIndices` and `testIndices` for each of the `nSplits` folds.
    public func split<T: Hashable>(targets: [T]) throws -> [(trainIndices: [Int], testIndices: [Int])] {
        guard !targets.isEmpty else {
            throw PreprocessingError.emptyInput
        }
        guard targets.count >= nSplits else {
            throw PreprocessingError.dimensionMismatch(expected: nSplits, got: targets.count)
        }
        
        let n = targets.count
        var classIndices: [T: [Int]] = [:]
        for (idx, target) in targets.enumerated() {
            classIndices[target, default: []].append(idx)
        }
        
        var rng: SeedableRandomNumberGenerator? = seed != nil ? SeedableRandomNumberGenerator(seed: seed!) : nil
        var foldTestIndices = [[Int]](repeating: [], count: nSplits)
        
        for (_, indices) in classIndices {
            var clsIdxs = indices
            if shuffle {
                if rng != nil {
                    clsIdxs.shuffle(using: &rng!)
                } else {
                    clsIdxs.shuffle()
                }
            }
            
            // Distribute class indices across folds as evenly as possible
            for (i, sampleIdx) in clsIdxs.enumerated() {
                let fold = i % nSplits
                foldTestIndices[fold].append(sampleIdx)
            }
        }
        
        var results: [(trainIndices: [Int], testIndices: [Int])] = []
        results.reserveCapacity(nSplits)
        
        let allIndicesSet = Set(0..<n)
        for fold in 0..<nSplits {
            var testIdxs = foldTestIndices[fold]
            if shuffle {
                if rng != nil {
                    testIdxs.shuffle(using: &rng!)
                } else {
                    testIdxs.shuffle()
                }
            }
            let testSet = Set(testIdxs)
            var trainIdxs = Array(allIndicesSet.subtracting(testSet)).sorted()
            if shuffle {
                if rng != nil {
                    trainIdxs.shuffle(using: &rng!)
                } else {
                    trainIdxs.shuffle()
                }
            }
            results.append((trainIndices: trainIdxs, testIndices: testIdxs))
        }
        
        return results
    }
}
