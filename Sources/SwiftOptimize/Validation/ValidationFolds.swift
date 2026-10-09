import Foundation
import SwiftML

// MARK: - Validation Error

/// Errors thrown by cross-validation splitters and validation routines.
public enum ValidationError: Error, LocalizedError, Sendable, Equatable {
    /// Number of splits is below the minimum required for the splitter.
    case invalidFoldCount(Int)
    /// Input dataset is empty.
    case emptyDataset
    /// Number of feature rows does not match number of targets.
    case dimensionMismatch(features: Int, targets: Int)
    /// Total number of samples is insufficient for the requested number of folds.
    case insufficientSamples(samples: Int, required: Int)
    /// A specific class stratum does not have enough samples to populate all folds.
    case insufficientClassSamples(label: Int, count: Int, requiredFolds: Int)
    /// Feature rows count does not match groups array length.
    case groupCountMismatch(features: Int, groups: Int)
    /// Number of unique groups is less than requested number of folds.
    case insufficientGroups(groups: Int, required: Int)
    /// A configuration parameter is invalid.
    case invalidParameter(String)

    public var errorDescription: String? {
        switch self {
        case .invalidFoldCount(let k):
            return "Invalid number of folds: \(k). Minimum required is 2 (or 1 for TimeSeriesSplit)."
        case .emptyDataset:
            return "Dataset cannot be empty for cross-validation."
        case .dimensionMismatch(let f, let t):
            return "Dimension mismatch: features count (\(f)) does not match targets count (\(t))."
        case .insufficientSamples(let s, let req):
            return "Insufficient samples: \(s) samples provided, but at least \(req) are required."
        case .insufficientClassSamples(let label, let count, let req):
            return "Class \(label) has only \(count) sample(s), but at least \(req) are required for StratifiedKFold."
        case .groupCountMismatch(let f, let g):
            return "Dimension mismatch: features count (\(f)) does not match groups count (\(g))."
        case .insufficientGroups(let g, let req):
            return "Insufficient groups: \(g) unique groups provided, but at least \(req) are required for GroupKFold."
        case .invalidParameter(let msg):
            return "Invalid validation parameter: \(msg)"
        }
    }
}

// MARK: - StratifiedKFold

/// Stratified K-Fold cross-validator.
/// Provides train/validation indices such that class proportions are preserved in each fold.
public struct StratifiedKFold: Sendable {
    /// The number of splits (folds).
    public let nSplits: Int
    /// Whether to shuffle samples within each class before splitting.
    public let shuffle: Bool
    /// Random seed for deterministic reproducibility.
    public let seed: Int

    /// Creates a new StratifiedKFold cross-validator.
    /// - Parameters:
    ///   - nSplits: Number of folds (must be at least 2).
    ///   - shuffle: Whether to shuffle samples within each stratum.
    ///   - seed: Random seed for shuffling.
    /// - Throws: `ValidationError.invalidFoldCount` if `nSplits < 2`.
    public init(nSplits: Int = 5, shuffle: Bool = true, seed: Int = 42) throws {
        guard nSplits >= 2 else {
            throw ValidationError.invalidFoldCount(nSplits)
        }
        self.nSplits = nSplits
        self.shuffle = shuffle
        self.seed = seed
    }

    /// Splits features and targets into K stratified train/validation folds.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors.
    ///   - targets: 1D array of target class labels.
    /// - Throws: `ValidationError` if data is empty, counts mismatch, or any class has fewer samples than `nSplits`.
    /// - Returns: An array of `nSplits` `Fold` instances.
    public func split(features: [[Double]], targets: [Double]) throws -> [Fold] {
        let n = features.count
        guard n > 0 else {
            throw ValidationError.emptyDataset
        }
        guard n == targets.count else {
            throw ValidationError.dimensionMismatch(features: n, targets: targets.count)
        }
        guard n >= nSplits else {
            throw ValidationError.insufficientSamples(samples: n, required: nSplits)
        }

        // Group sample indices by integer class label
        var classIndices: [Int: [Int]] = [:]
        for (i, target) in targets.enumerated() {
            let label = Int(target.rounded())
            classIndices[label, default: []].append(i)
        }

        // Sort class labels deterministically to eliminate Dictionary hash-seed variation across processes
        let sortedLabels = classIndices.keys.sorted()

        var rng = SeededRandom(seed: seed)

        // Shuffle indices per class in deterministic order if requested, and validate stratum counts
        var processedClassIndices: [(label: Int, indices: [Int])] = []
        processedClassIndices.reserveCapacity(sortedLabels.count)

        for label in sortedLabels {
            var idxs = classIndices[label]!
            guard idxs.count >= nSplits else {
                throw ValidationError.insufficientClassSamples(label: label, count: idxs.count, requiredFolds: nSplits)
            }
            if shuffle {
                for i in stride(from: idxs.count - 1, through: 1, by: -1) {
                    let j = rng.nextInt(upperBound: i + 1)
                    idxs.swapAt(i, j)
                }
            }
            processedClassIndices.append((label, idxs))
        }

        // Allocate samples from each class across the K folds in a balanced cyclic manner
        var foldValIndices = [[Int]](repeating: [], count: nSplits)
        var foldOffset = 0
        for (_, idxs) in processedClassIndices {
            for (i, idx) in idxs.enumerated() {
                let foldIdx = (foldOffset + i) % nSplits
                foldValIndices[foldIdx].append(idx)
            }
            foldOffset = (foldOffset + idxs.count) % nSplits
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
    /// Number of splits.
    public let nSplits: Int
    /// Maximum size for a single training set.
    public let maxTrainSize: Int?

    /// Creates a new TimeSeriesSplit instance.
    /// - Parameters:
    ///   - nSplits: Number of splits (must be at least 1).
    ///   - maxTrainSize: Optional maximum training set size.
    /// - Throws: `ValidationError` if parameters are invalid.
    public init(nSplits: Int = 5, maxTrainSize: Int? = nil) throws {
        guard nSplits >= 1 else {
            throw ValidationError.invalidFoldCount(nSplits)
        }
        if let maxTrain = maxTrainSize {
            guard maxTrain > 0 else {
                throw ValidationError.invalidParameter("maxTrainSize must be positive")
            }
        }
        self.nSplits = nSplits
        self.maxTrainSize = maxTrainSize
    }

    /// Splits features and targets into expanding time series train/val folds.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors.
    ///   - targets: 1D array of target values.
    /// - Throws: `ValidationError` if data is empty, mismatched, or has fewer samples than `nSplits + 1`.
    /// - Returns: An array of `nSplits` `Fold` instances.
    public func split(features: [[Double]], targets: [Double]) throws -> [Fold] {
        let n = features.count
        guard n > 0 else {
            throw ValidationError.emptyDataset
        }
        guard n == targets.count else {
            throw ValidationError.dimensionMismatch(features: n, targets: targets.count)
        }
        guard n > nSplits else {
            throw ValidationError.insufficientSamples(samples: n, required: nSplits + 1)
        }

        let testSize = n / (nSplits + 1)
        var folds = [Fold]()
        folds.reserveCapacity(nSplits)

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
    /// Number of splits.
    public let nSplits: Int

    /// Creates a new GroupKFold instance.
    /// - Parameters:
    ///   - nSplits: Number of splits (must be at least 2).
    /// - Throws: `ValidationError.invalidFoldCount` if `nSplits < 2`.
    public init(nSplits: Int = 5) throws {
        guard nSplits >= 2 else {
            throw ValidationError.invalidFoldCount(nSplits)
        }
        self.nSplits = nSplits
    }

    /// Splits data into K group-preserving folds.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors.
    ///   - targets: 1D array of target values.
    ///   - groups: 1D array of integer group labels per sample.
    /// - Throws: `ValidationError` if data is invalid or unique groups count is less than `nSplits`.
    /// - Returns: An array of `nSplits` `Fold` instances.
    public func split(features: [[Double]], targets: [Double], groups: [Int]) throws -> [Fold] {
        let n = features.count
        guard n > 0 else {
            throw ValidationError.emptyDataset
        }
        guard n == targets.count else {
            throw ValidationError.dimensionMismatch(features: n, targets: targets.count)
        }
        guard n == groups.count else {
            throw ValidationError.groupCountMismatch(features: n, groups: groups.count)
        }

        let uniqueGroups = Array(Set(groups)).sorted()
        let numGroups = uniqueGroups.count
        guard numGroups >= nSplits else {
            throw ValidationError.insufficientGroups(groups: numGroups, required: nSplits)
        }

        let k = nSplits
        var groupFolds: [Int: Int] = [:]
        for (i, group) in uniqueGroups.enumerated() {
            groupFolds[group] = i % k
        }

        var folds = [Fold]()
        folds.reserveCapacity(k)
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

