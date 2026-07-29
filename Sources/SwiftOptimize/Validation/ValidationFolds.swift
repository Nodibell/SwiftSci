import Foundation
import SwiftML


// MARK: - StratifiedKFold

/// Stratified K-Fold cross-validator.
/// Provides train/validation indices such that class proportions are preserved in each fold.
public struct StratifiedKFold: Sendable {
    /// The n splits.
    public let nSplits: Int
    /// The shuffle.
    public let shuffle: Bool
    /// The seed.
    public let seed: Int

    /// Creates a new instance.
    /// - Parameters:
    ///   - nSplits: The n splits.
    ///   - shuffle: The shuffle.
    ///   - seed: The seed.
    public init(nSplits: Int = 5, shuffle: Bool = true, seed: Int = 42) {
        precondition(nSplits >= 2, "StratifiedKFold requires at least 2 splits")
        self.nSplits = nSplits
        self.shuffle = shuffle
        self.seed = seed
    }

    /// Split.
    /// - Parameters:
    ///   - features: The features.
    ///   - targets: The targets.
    /// - Returns: A `[Fold]` result.
    public func split(features: [[Double]], targets: [Double]) -> [Fold] {
        let n = features.count
        guard n > 0 else { return [] }

        // Group sample indices by integer class label
        var classIndices: [Int: [Int]] = [:]
        for (i, target) in targets.enumerated() {
            let label = Int(target)
            classIndices[label, default: []].append(i)
        }

        var rng = SeededRandom(seed: seed)

        // Shuffle indices per class if requested
        for (label, idxs) in classIndices {
            var shuffled = idxs
            if shuffle {
                for i in stride(from: shuffled.count - 1, through: 1, by: -1) {
                    let j = rng.nextInt(upperBound: i + 1)
                    shuffled.swapAt(i, j)
                }
            }
            classIndices[label] = shuffled
        }

        // Allocate samples from each class across the K folds
        var foldValIndices = [[Int]](repeating: [], count: nSplits)
        for (_, idxs) in classIndices {
            for (i, idx) in idxs.enumerated() {
                let foldIdx = i % nSplits
                foldValIndices[foldIdx].append(idx)
            }
        }

        var folds = [Fold]()
        folds.reserveCapacity(nSplits)

        for k in 0..<nSplits {
            let valSet = Set(foldValIndices[k])
            var trainIdx = [Int]()
            var valIdx = [Int]()

            for i in 0..<n {
                if valSet.contains(i) {
                    valIdx.append(i)
                } else {
                    trainIdx.append(i)
                }
            }

            folds.append(Fold(
                trainFeatures: trainIdx.map { features[$0] },
                trainTargets: trainIdx.map { targets[$0] },
                valFeatures: valIdx.map { features[$0] },
                valTargets: valIdx.map { targets[$0] }
            ))
        }

        return folds
    }
}

// MARK: - TimeSeriesSplit

/// Time Series cross-validator.
/// Provides train/validation splits where training set expands over time to prevent data leakage.
public struct TimeSeriesSplit: Sendable {
    /// The n splits.
    public let nSplits: Int
    /// The max train size.
    public let maxTrainSize: Int?

    /// Creates a new instance.
    /// - Parameters:
    ///   - nSplits: The n splits.
    ///   - maxTrainSize: The max train size.
    public init(nSplits: Int = 5, maxTrainSize: Int? = nil) {
        precondition(nSplits >= 1, "TimeSeriesSplit requires at least 1 split")
        self.nSplits = nSplits
        self.maxTrainSize = maxTrainSize
    }

    /// Split.
    /// - Parameters:
    ///   - features: The features.
    ///   - targets: The targets.
    /// - Returns: A `[Fold]` result.
    public func split(features: [[Double]], targets: [Double]) -> [Fold] {
        let n = features.count
        guard n > nSplits else { return [] }

        let testSize = n / (nSplits + 1)
        var folds = [Fold]()

        for i in 0..<nSplits {
            let valStart = n - (nSplits - i) * testSize
            let valEnd = valStart + testSize

            var trainStart = 0
            if let maxTrain = maxTrainSize, valStart - maxTrain > 0 {
                trainStart = valStart - maxTrain
            }

            let trainFeatures = Array(features[trainStart..<valStart])
            let trainTargets = Array(targets[trainStart..<valStart])
            let valFeatures = Array(features[valStart..<valEnd])
            let valTargets = Array(targets[valStart..<valEnd])

            folds.append(Fold(
                trainFeatures: trainFeatures,
                trainTargets: trainTargets,
                valFeatures: valFeatures,
                valTargets: valTargets
            ))
        }

        return folds
    }
}

// MARK: - GroupKFold

/// Group K-Fold cross-validator.
/// Ensures identical group identifiers do not appear in both training and validation splits.
public struct GroupKFold: Sendable {
    /// The n splits.
    public let nSplits: Int

    /// Creates a new instance.
    /// - Parameters:
    ///   - nSplits: The n splits.
    public init(nSplits: Int = 5) {
        precondition(nSplits >= 2, "GroupKFold requires at least 2 splits")
        self.nSplits = nSplits
    }

    /// Split.
    /// - Parameters:
    ///   - features: The features.
    ///   - targets: The targets.
    ///   - groups: The groups.
    /// - Returns: A `[Fold]` result.
    public func split(features: [[Double]], targets: [Double], groups: [Int]) -> [Fold] {
        let n = features.count
        guard n == groups.count else { return [] }

        let uniqueGroups = Array(Set(groups)).sorted()
        let numGroups = uniqueGroups.count
        let k = min(nSplits, numGroups)

        var groupFolds: [Int: Int] = [:]
        for (i, group) in uniqueGroups.enumerated() {
            groupFolds[group] = i % k
        }

        var folds = [Fold]()
        for foldIdx in 0..<k {
            var trainIdx = [Int]()
            var valIdx = [Int]()

            for i in 0..<n {
                let g = groups[i]
                if groupFolds[g] == foldIdx {
                    valIdx.append(i)
                } else {
                    trainIdx.append(i)
                }
            }

            folds.append(Fold(
                trainFeatures: trainIdx.map { features[$0] },
                trainTargets: trainIdx.map { targets[$0] },
                valFeatures: valIdx.map { features[$0] },
                valTargets: valIdx.map { targets[$0] }
            ))
        }

        return folds
    }
}
