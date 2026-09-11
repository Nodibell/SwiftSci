import Foundation
import SwiftPreprocessing

// MARK: - Bootstrap Helper

// internal: not part of the public API; used by both Classifier and Regressor
func bootstrapSample(features: [[Double]], targets: [Double], seed: Int, maxSamples: Int? = nil) -> ([[Double]], [Double]) {
    let n = features.count
    let sampleSize = min(n, maxSamples ?? n)
    var rng = SeededRandom(seed: seed)
    var sampledFeatures = [[Double]]()
    var sampledTargets = [Double]()
    sampledFeatures.reserveCapacity(sampleSize)
    sampledTargets.reserveCapacity(sampleSize)
    for _ in 0..<sampleSize {
        let i = rng.nextInt(upperBound: n)
        sampledFeatures.append(features[i])
        sampledTargets.append(targets[i])
    }
    return (sampledFeatures, sampledTargets)
}

// MARK: - Random Forest Classifier

/// Actor-isolated Random Forest Classifier.
/// Each tree is trained concurrently in a TaskGroup on a bootstrapped sample.
public actor RandomForestClassifier: ClassifierEstimator {
    /// The n estimators.
    public let nEstimators: Int
    /// The max depth.
    public let maxDepth: Int
    /// The max features.
    public let maxFeatures: Int?
    /// The max samples per tree.
    public let maxSamples: Int?
    /// The min samples split.
    public let minSamplesSplit: Int
    /// The criterion.
    public let criterion: SplitCriterion
    /// Optional random state used to seed tree bootstrapping and feature selection deterministically.
    public let randomState: Int?

    // DOD Architecture: the forest is stored as an array of flat node arrays
    private var trees: [[FlatTreeNode]] = []
    
    /// The collection of fitted trees as flat node arrays.
    public var flatTrees: [[FlatTreeNode]] { trees }
    
    private var numClasses: Int = 0
    private var numFeatures: Int = 0

    /// The feature importances.
    public var featureImportances: [Double]? {
        guard numFeatures > 0, !trees.isEmpty else { return nil }
        var aggregated = [Double](repeating: 0.0, count: numFeatures)
        var validTrees = 0

        for tree in trees {
            if let treeImp = computeFeatureImportances(nodes: tree, numFeatures: numFeatures) {
                for i in 0..<numFeatures {
                    aggregated[i] += treeImp[i]
                }
                validTrees += 1
            }
        }

        guard validTrees > 0 else { return nil }
        let total = aggregated.reduce(0.0, +)
        if total > 0 {
            return aggregated.map { $0 / total }
        }
        return aggregated
    }

    /// Creates a new random forest classifier.
    /// - Parameters:
    ///   - nEstimators: The number of trees in the forest. Must be greater than 0.
    ///   - maxDepth: The maximum depth of each decision tree.
    ///   - maxFeatures: The number of features to consider when looking for the best split. Defaults to `sqrt(numFeatures)`.
    ///   - maxSamples: Optional maximum number of bootstrap samples per tree (prevents OOM on large datasets).
    ///   - minSamplesSplit: The minimum number of samples required to split an internal node.
    ///   - criterion: The function to measure the quality of a split (.gini or .entropy).
    ///   - randomState: Controls the randomness of the bootstrapping of samples and tree building.
    /// - Throws: `SwiftMLError.invalidParameter` if parameters are invalid.
    ///
    /// ## Thread Safety
    /// Actor-isolated model. Safe to initialize and access across concurrent execution contexts.
    public init(
        nEstimators: Int = 100,
        maxDepth: Int = 10,
        maxFeatures: Int? = nil,
        maxSamples: Int? = nil,
        minSamplesSplit: Int = 2,
        criterion: SplitCriterion = .gini,
        randomState: Int? = nil
    ) throws {
        guard nEstimators > 0 else { throw SwiftMLError.invalidParameter("nEstimators must be > 0") }
        self.nEstimators = nEstimators
        self.maxDepth = maxDepth
        self.maxFeatures = maxFeatures
        self.maxSamples = maxSamples
        self.minSamplesSplit = minSamplesSplit
        self.criterion = criterion
        self.randomState = randomState
    }

    /// Fit classifier without progress callback.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - targets: 1D array of ground-truth target values of length `N`.
    /// - Throws: `SwiftMLError` if feature-target dimensions mismatch, inputs are empty, or optimization fails.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        try await fit(features: features, targets: targets, onProgress: nil)
    }

    /// Fits the random forest classifier using concurrent tree construction.
    /// - Parameters:
    ///   - features: Matrix of shape `[n_samples, n_features]`.
    ///   - targets: Array of class labels of shape `[n_samples]`.
    ///   - onProgress: Optional progress callback reporting `(completedTrees, totalTrees)`.
    /// - Throws: `SwiftMLError` if features are empty or dimension mismatch occurs.
    ///
    /// ## Concurrency
    /// Uses `withThrowingTaskGroup` to construct decision trees in parallel across all available CPU cores.
    public func fit(
        features: [[Double]],
        targets: [Double],
        onProgress: (@Sendable (Int, Int) -> Void)?
    ) async throws {
        guard !features.isEmpty else { throw SwiftMLError.emptyInput }
        guard features.count == targets.count else {
            throw SwiftMLError.dimensionMismatch(expected: features.count, got: targets.count)
        }

        numClasses = Int(targets.max() ?? 0) + 1
        numFeatures = features[0].count
        let maxDepth = self.maxDepth
        let maxFeatures = self.maxFeatures ?? max(1, Int(sqrt(Double(numFeatures))))
        let maxSamples = self.maxSamples ?? (features.count > 10_000 ? 10_000 : features.count)
        let minSamplesSplit = self.minSamplesSplit
        let criterion = self.criterion
        let nEstimators = self.nEstimators
        let baseSeed = self.randomState ?? 42

        let trainedTrees: [[FlatTreeNode]] = try await withThrowingTaskGroup(of: (Int, [FlatTreeNode]).self) { group in
            for i in 0..<nEstimators {
                let treeSeed = baseSeed &+ (i &* 10007)
                group.addTask {
                    let (bX, bY) = bootstrapSample(features: features, targets: targets, seed: treeSeed, maxSamples: maxSamples)
                    let presorted = createPresortedIndices(X: bX)
                    var nodes = [FlatTreeNode]()
                    _ = RandomForestClassifier.buildTreeSync(
                        X: bX, y: bY,
                        indices: Array(0..<bX.count),
                        presortedIndices: presorted,
                        depth: 0,
                        maxDepth: maxDepth,
                        minSamplesSplit: minSamplesSplit,
                        criterion: criterion,
                        maxFeatures: maxFeatures,
                        seed: treeSeed,
                        nodes: &nodes
                    )
                    return (i, nodes)
                }
            }

            var indexedTrees = [(Int, [FlatTreeNode])]()
            indexedTrees.reserveCapacity(nEstimators)
            var doneCount = 0
            for try await (idx, treeNodes) in group {
                indexedTrees.append((idx, treeNodes))
                doneCount += 1
                onProgress?(doneCount, nEstimators)
            }
            indexedTrees.sort(by: { $0.0 < $1.0 })
            return indexedTrees.map { $0.1 }
        }

        self.trees = trainedTrees
    }

    /// Predict.
    /// - Parameters:
    ///   - features: The features.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[Int]` result.
    public func predict(features: [[Double]]) async throws -> [Int] {
        guard !trees.isEmpty else { throw SwiftMLError.notFitted }
        return features.map { sample in
            var votes = [Int: Int]()
            for treeNodes in trees {
                let pred = Int(RandomForestClassifier.predictSample(sample, nodes: treeNodes))
                votes[pred, default: 0] += 1
            }
            return votes.sorted(by: { if $0.value != $1.value { return $0.value > $1.value } else { return $0.key < $1.key } }).first?.key ?? 0
        }
    }

    /// Predict probability.
    /// - Parameters:
    ///   - features: The features.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[[Double]]` result.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        guard !trees.isEmpty else { throw SwiftMLError.notFitted }
        return features.map { sample in
            var votes = [Double](repeating: 0, count: numClasses)
            for treeNodes in trees {
                let pred = Int(RandomForestClassifier.predictSample(sample, nodes: treeNodes))
                if pred < votes.count {
                    votes[pred] += 1.0
                }
            }
            let total = Double(trees.count)
            return votes.map { $0 / total }
        }
    }

    /// Returns the fitted forest as an array of per-tree `FlatTreeNode` arrays.
    ///
    /// Used by ``CoreMLExportable`` conformance to build binary `.mlmodel` artifacts.
    /// - Returns: All tree node arrays, or an empty array if the model is not fitted.
    public func getForestTrees() -> [[FlatTreeNode]] { trees }

    /// Returns the number of input features seen during training.
    /// - Returns: Feature count, or `0` if the model is not fitted.
    public func getNumFeatures() -> Int { numFeatures }

    // MARK: Static helpers (no actor isolation needed)

    private static func buildTreeSync(
        X: [[Double]],
        y: [Double],
        indices: [Int],
        presortedIndices: [[Int]],
        depth: Int,
        maxDepth: Int,
        minSamplesSplit: Int,
        criterion: SplitCriterion,
        maxFeatures: Int?,
        seed: Int = 42,
        nodes: inout [FlatTreeNode]
    ) -> Int {
        let labels = indices.map { y[$0] }
        let majority = labels.mostFrequent()

        if depth >= maxDepth || indices.count < minSamplesSplit || Set(labels).count == 1 {
            nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: majority, isLeaf: true, impurityGain: 0.0))
            return nodes.count - 1
        }

        let splitSeed = seed &+ (depth &* 31) &+ (indices.count &* 101)
        guard let split = bestSplit(X: X, y: y, indices: indices, presortedIndices: presortedIndices, criterion: criterion, maxFeatures: maxFeatures, seed: splitSeed) else {
            nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: majority, isLeaf: true, impurityGain: 0.0))
            return nodes.count - 1
        }

        let currentIndex = nodes.count
        nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: 0, isLeaf: false, impurityGain: 0.0))

        let leftIndex  = buildTreeSync(X: X, y: y, indices: split.leftIndices,  presortedIndices: presortedIndices, depth: depth + 1, maxDepth: maxDepth, minSamplesSplit: minSamplesSplit, criterion: criterion, maxFeatures: maxFeatures, seed: seed &* 3 &+ 1, nodes: &nodes)
        let rightIndex = buildTreeSync(X: X, y: y, indices: split.rightIndices, presortedIndices: presortedIndices, depth: depth + 1, maxDepth: maxDepth, minSamplesSplit: minSamplesSplit, criterion: criterion, maxFeatures: maxFeatures, seed: seed &* 3 &+ 2, nodes: &nodes)

        let nodeGain = split.gain * Double(indices.count)
        nodes[currentIndex] = FlatTreeNode(
            featureIndex: split.featureIndex,
            threshold: split.threshold,
            leftChild: leftIndex,
            rightChild: rightIndex,
            value: majority,
            isLeaf: false,
            impurityGain: nodeGain
        )

        return currentIndex
    }

    private static func predictSample(_ x: [Double], nodes: [FlatTreeNode]) -> Double {
        guard !nodes.isEmpty else { return 0 }
        var curr = 0
        while !nodes[curr].isLeaf {
            let node = nodes[curr]
            let val = (node.featureIndex >= 0 && node.featureIndex < x.count) ? x[node.featureIndex] : 0.0
            if val <= node.threshold {
                curr = node.leftChild
            } else {
                curr = node.rightChild
            }
        }
        return nodes[curr].value
    }
}

// MARK: - Random Forest Regressor

/// Actor-isolated Random Forest Regressor.
/// Each tree is trained concurrently in a TaskGroup on a bootstrapped sample.
public actor RandomForestRegressor: RegressorEstimator {
    /// The n estimators.
    public let nEstimators: Int
    /// The max depth.
    public let maxDepth: Int
    /// The max features.
    public let maxFeatures: Int?
    /// The max samples per tree.
    public let maxSamples: Int?
    /// The min samples split.
    public let minSamplesSplit: Int
    /// Optional random state used to seed tree bootstrapping and feature selection deterministically.
    public let randomState: Int?

    private var trees: [[FlatTreeNode]] = []
    
    /// The collection of fitted trees as flat node arrays.
    public var flatTrees: [[FlatTreeNode]] { trees }
    
    private var numFeatures: Int = 0

    /// The feature importances.
    public var featureImportances: [Double]? {
        guard numFeatures > 0, !trees.isEmpty else { return nil }
        var aggregated = [Double](repeating: 0.0, count: numFeatures)
        var validTrees = 0

        for tree in trees {
            if let treeImp = computeFeatureImportances(nodes: tree, numFeatures: numFeatures) {
                for i in 0..<numFeatures {
                    aggregated[i] += treeImp[i]
                }
                validTrees += 1
            }
        }

        guard validTrees > 0 else { return nil }
        let total = aggregated.reduce(0.0, +)
        if total > 0 {
            return aggregated.map { $0 / total }
        }
        return aggregated
    }

    /// Creates a new random forest regressor.
    /// - Parameters:
    ///   - nEstimators: The number of trees in the forest. Must be greater than 0.
    ///   - maxDepth: The maximum depth of each decision tree.
    ///   - maxFeatures: The number of features to consider when looking for the best split. Defaults to `sqrt(numFeatures)`.
    ///   - maxSamples: Optional maximum number of bootstrap samples per tree.
    ///   - minSamplesSplit: The minimum number of samples required to split an internal node.
    ///   - randomState: Controls the randomness of the bootstrapping of samples and tree building.
    /// - Throws: `SwiftMLError.invalidParameter` if parameters are invalid.
    ///
    /// ## Thread Safety
    /// Actor-isolated model. Safe to initialize and access across concurrent execution contexts.
    public init(
        nEstimators: Int = 100,
        maxDepth: Int = 10,
        maxFeatures: Int? = nil,
        maxSamples: Int? = nil,
        minSamplesSplit: Int = 2,
        randomState: Int? = nil
    ) throws {
        guard nEstimators > 0 else { throw SwiftMLError.invalidParameter("nEstimators must be > 0") }
        self.nEstimators = nEstimators
        self.maxDepth = maxDepth
        self.maxFeatures = maxFeatures
        self.maxSamples = maxSamples
        self.minSamplesSplit = minSamplesSplit
        self.randomState = randomState
    }

    /// Fits the regressor model.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - targets: 1D array of ground-truth target values of length `N`.
    /// - Throws: `SwiftMLError` if feature-target dimensions mismatch, inputs are empty, or optimization fails.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        try await fit(features: features, targets: targets, onProgress: nil)
    }

    /// Fits the random forest regressor using concurrent tree construction.
    /// - Parameters:
    ///   - features: Matrix of shape `[n_samples, n_features]`.
    ///   - targets: Continuous target array of shape `[n_samples]`.
    ///   - onProgress: Optional progress callback reporting `(completed, total)`.
    /// - Throws: `SwiftMLError` if features are empty or dimension mismatch occurs.
    ///
    /// ## Concurrency
    /// Uses `withThrowingTaskGroup` to construct decision trees in parallel across all available CPU cores.
    public func fit(
        features: [[Double]],
        targets: [Double],
        onProgress: (@Sendable (Int, Int) -> Void)?
    ) async throws {
        guard !features.isEmpty else { throw SwiftMLError.emptyInput }
        guard features.count == targets.count else {
            throw SwiftMLError.dimensionMismatch(expected: features.count, got: targets.count)
        }

        numFeatures = features[0].count
        let maxDepth = self.maxDepth
        let maxFeatures = self.maxFeatures ?? max(1, Int(sqrt(Double(numFeatures))))
        let maxSamples = self.maxSamples ?? (features.count > 10_000 ? 10_000 : features.count)
        let minSamplesSplit = self.minSamplesSplit
        let nEstimators = self.nEstimators
        let baseSeed = self.randomState ?? 42

        let trainedTrees: [[FlatTreeNode]] = try await withThrowingTaskGroup(of: (Int, [FlatTreeNode]).self) { group in
            for i in 0..<nEstimators {
                let treeSeed = baseSeed &+ (i &* 10007)
                group.addTask {
                    let (bX, bY) = bootstrapSample(features: features, targets: targets, seed: treeSeed, maxSamples: maxSamples)
                    let presorted = createPresortedIndices(X: bX)
                    var nodes = [FlatTreeNode]()
                    _ = RandomForestRegressor.buildTreeSync(
                        X: bX, y: bY,
                        indices: Array(0..<bX.count),
                        presortedIndices: presorted,
                        depth: 0,
                        maxDepth: maxDepth,
                        minSamplesSplit: minSamplesSplit,
                        maxFeatures: maxFeatures,
                        seed: treeSeed,
                        nodes: &nodes
                    )
                    return (i, nodes)
                }
            }
            var indexedTrees = [(Int, [FlatTreeNode])]()
            indexedTrees.reserveCapacity(nEstimators)
            var doneCount = 0
            for try await (idx, treeNodes) in group {
                indexedTrees.append((idx, treeNodes))
                doneCount += 1
                onProgress?(doneCount, nEstimators)
            }
            indexedTrees.sort(by: { $0.0 < $1.0 })
            return indexedTrees.map { $0.1 }
        }

        self.trees = trainedTrees
    }

    /// Predict.
    /// - Parameters:
    ///   - features: The features.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[Double]` result.
    public func predict(features: [[Double]]) async throws -> [Double] {
        guard !trees.isEmpty else { throw SwiftMLError.notFitted }
        return features.map { sample in
            let preds = trees.map { RandomForestRegressor.predictSample(sample, nodes: $0) }
            return preds.mean()
        }
    }

    /// Returns the fitted forest as an array of per-tree `FlatTreeNode` arrays.
    ///
    /// Used by ``CoreMLExportable`` conformance to build binary `.mlmodel` artifacts.
    /// - Returns: All tree node arrays, or an empty array if the model is not fitted.
    public func getForestTrees() -> [[FlatTreeNode]] { trees }

    /// Returns the number of input features seen during training.
    /// - Returns: Feature count, or `0` if the model is not fitted.
    public func getNumFeatures() -> Int { numFeatures }

    private static func buildTreeSync(
        X: [[Double]],
        y: [Double],
        indices: [Int],
        presortedIndices: [[Int]],
        depth: Int,
        maxDepth: Int,
        minSamplesSplit: Int,
        maxFeatures: Int?,
        seed: Int = 42,
        nodes: inout [FlatTreeNode]
    ) -> Int {
        let values = indices.map { y[$0] }
        let mean = values.mean()

        if depth >= maxDepth || indices.count < minSamplesSplit {
            nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: mean, isLeaf: true, impurityGain: 0.0))
            return nodes.count - 1
        }

        let splitSeed = seed &+ (depth &* 31) &+ (indices.count &* 101)
        guard let split = bestSplit(X: X, y: y, indices: indices, presortedIndices: presortedIndices, criterion: .mse, maxFeatures: maxFeatures, seed: splitSeed),
              split.gain > 0 else {
            nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: mean, isLeaf: true, impurityGain: 0.0))
            return nodes.count - 1
        }

        let currentIndex = nodes.count
        nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: 0, isLeaf: false, impurityGain: 0.0))

        let left  = buildTreeSync(X: X, y: y, indices: split.leftIndices,  presortedIndices: presortedIndices, depth: depth + 1, maxDepth: maxDepth, minSamplesSplit: minSamplesSplit, maxFeatures: maxFeatures, seed: seed &* 3 &+ 1, nodes: &nodes)
        let right = buildTreeSync(X: X, y: y, indices: split.rightIndices, presortedIndices: presortedIndices, depth: depth + 1, maxDepth: maxDepth, minSamplesSplit: minSamplesSplit, maxFeatures: maxFeatures, seed: seed &* 3 &+ 2, nodes: &nodes)

        let nodeGain = split.gain * Double(indices.count)
        nodes[currentIndex] = FlatTreeNode(
            featureIndex: split.featureIndex,
            threshold: split.threshold,
            leftChild: left,
            rightChild: right,
            value: mean,
            isLeaf: false,
            impurityGain: nodeGain
        )

        return currentIndex
    }

    private static func predictSample(_ x: [Double], nodes: [FlatTreeNode]) -> Double {
        guard !nodes.isEmpty else { return 0 }
        var curr = 0

        while !nodes[curr].isLeaf {
            let node = nodes[curr]
            let val = (node.featureIndex >= 0 && node.featureIndex < x.count) ? x[node.featureIndex] : 0.0
            if val <= node.threshold {
                curr = node.leftChild
            } else {
                curr = node.rightChild
            }
        }
        return nodes[curr].value
    }
}
