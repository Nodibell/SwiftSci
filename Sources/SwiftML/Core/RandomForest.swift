import Foundation
import SwiftPreprocessing

// MARK: - Bootstrap Helper

// internal: not part of the public API; used by both Classifier and Regressor
func bootstrapSample(features: [[Double]], targets: [Double], seed: Int) -> ([[Double]], [Double]) {
    let n = features.count
    var rng = SeededRandom(seed: seed)
    var sampledFeatures = [[Double]]()
    var sampledTargets = [Double]()
    sampledFeatures.reserveCapacity(n)
    sampledTargets.reserveCapacity(n)
    for _ in 0..<n {
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
    public let nEstimators: Int
    public let maxDepth: Int
    public let maxFeatures: Int?
    public let minSamplesSplit: Int
    public let criterion: SplitCriterion

    // DOD Architecture: the forest is stored as an array of flat node arrays
    private var trees: [[FlatTreeNode]] = []
    private var numClasses: Int = 0
    private var numFeatures: Int = 0

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

    public init(
        nEstimators: Int = 100,
        maxDepth: Int = 10,
        maxFeatures: Int? = nil,
        minSamplesSplit: Int = 2,
        criterion: SplitCriterion = .gini
    ) throws {
        guard nEstimators > 0 else { throw MLError.invalidParameter("nEstimators must be > 0") }
        self.nEstimators = nEstimators
        self.maxDepth = maxDepth
        self.maxFeatures = maxFeatures
        self.minSamplesSplit = minSamplesSplit
        self.criterion = criterion
    }

    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty else { throw MLError.emptyInput }
        guard features.count == targets.count else {
            throw MLError.dimensionMismatch(expected: features.count, got: targets.count)
        }

        numClasses = Int(targets.max() ?? 0) + 1
        numFeatures = features[0].count
        let maxDepth = self.maxDepth
        let maxFeatures = self.maxFeatures
        let minSamplesSplit = self.minSamplesSplit
        let criterion = self.criterion

        let trainedTrees: [[FlatTreeNode]] = try await withThrowingTaskGroup(of: [FlatTreeNode].self) { group in
            for i in 0..<nEstimators {
                group.addTask {
                    let (bX, bY) = bootstrapSample(features: features, targets: targets, seed: i)
                    var nodes = [FlatTreeNode]()
                    _ = RandomForestClassifier.buildTreeSync(
                        X: bX, y: bY,
                        indices: Array(0..<bX.count),
                        depth: 0,
                        maxDepth: maxDepth,
                        minSamplesSplit: minSamplesSplit,
                        criterion: criterion,
                        maxFeatures: maxFeatures,
                        nodes: &nodes
                    )
                    return nodes
                }
            }
            var result = [[FlatTreeNode]]()
            for try await treeNodes in group {
                result.append(treeNodes)
            }
            return result
        }

        self.trees = trainedTrees
    }

    public func predict(features: [[Double]]) async throws -> [Int] {
        guard !trees.isEmpty else { throw MLError.notFitted }
        return features.map { sample in
            var votes = [Int: Int]()
            for treeNodes in trees {
                let pred = Int(RandomForestClassifier.predictSample(sample, nodes: treeNodes))
                votes[pred, default: 0] += 1
            }
            return votes.max(by: { $0.value < $1.value })?.key ?? 0
        }
    }

    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        guard !trees.isEmpty else { throw MLError.notFitted }
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

    // MARK: Static helpers (no actor isolation needed)

    private static func buildTreeSync(
        X: [[Double]],
        y: [Double],
        indices: [Int],
        depth: Int,
        maxDepth: Int,
        minSamplesSplit: Int,
        criterion: SplitCriterion,
        maxFeatures: Int?,
        nodes: inout [FlatTreeNode]
    ) -> Int {
        let labels = indices.map { y[$0] }
        let majority = labels.mostFrequent()

        if depth >= maxDepth || indices.count < minSamplesSplit || Set(labels).count == 1 {
            nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: majority, isLeaf: true, impurityGain: 0.0))
            return nodes.count - 1
        }

        guard let split = bestSplit(X: X, y: y, indices: indices, criterion: criterion, maxFeatures: maxFeatures) else {
            nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: majority, isLeaf: true, impurityGain: 0.0))
            return nodes.count - 1
        }

        let currentIndex = nodes.count
        nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: 0, isLeaf: false, impurityGain: 0.0))

        let leftIndex  = buildTreeSync(X: X, y: y, indices: split.leftIndices,  depth: depth + 1, maxDepth: maxDepth, minSamplesSplit: minSamplesSplit, criterion: criterion, maxFeatures: maxFeatures, nodes: &nodes)
        let rightIndex = buildTreeSync(X: X, y: y, indices: split.rightIndices, depth: depth + 1, maxDepth: maxDepth, minSamplesSplit: minSamplesSplit, criterion: criterion, maxFeatures: maxFeatures, nodes: &nodes)

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
            if x[node.featureIndex] <= node.threshold {
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
    public let nEstimators: Int
    public let maxDepth: Int
    public let maxFeatures: Int?
    public let minSamplesSplit: Int

    private var trees: [[FlatTreeNode]] = []
    private var numFeatures: Int = 0

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

    public init(
        nEstimators: Int = 100,
        maxDepth: Int = 10,
        maxFeatures: Int? = nil,
        minSamplesSplit: Int = 2
    ) throws {
        guard nEstimators > 0 else { throw MLError.invalidParameter("nEstimators must be > 0") }
        self.nEstimators = nEstimators
        self.maxDepth = maxDepth
        self.maxFeatures = maxFeatures
        self.minSamplesSplit = minSamplesSplit
    }

    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty else { throw MLError.emptyInput }
        guard features.count == targets.count else {
            throw MLError.dimensionMismatch(expected: features.count, got: targets.count)
        }

        numFeatures = features[0].count
        let maxDepth = self.maxDepth
        let maxFeatures = self.maxFeatures
        let minSamplesSplit = self.minSamplesSplit

        let trainedTrees: [[FlatTreeNode]] = try await withThrowingTaskGroup(of: [FlatTreeNode].self) { group in
            for i in 0..<nEstimators {
                group.addTask {
                    let (bX, bY) = bootstrapSample(features: features, targets: targets, seed: i)
                    var nodes = [FlatTreeNode]()
                    _ = RandomForestRegressor.buildTreeSync(
                        X: bX, y: bY,
                        indices: Array(0..<bX.count),
                        depth: 0,
                        maxDepth: maxDepth,
                        minSamplesSplit: minSamplesSplit,
                        maxFeatures: maxFeatures,
                        nodes: &nodes
                    )
                    return nodes
                }
            }
            var result = [[FlatTreeNode]]()
            for try await treeNodes in group {
                result.append(treeNodes)
            }
            return result
        }

        self.trees = trainedTrees
    }

    public func predict(features: [[Double]]) async throws -> [Double] {
        guard !trees.isEmpty else { throw MLError.notFitted }
        return features.map { sample in
            let preds = trees.map { RandomForestRegressor.predictSample(sample, nodes: $0) }
            return preds.mean()
        }
    }

    private static func buildTreeSync(
        X: [[Double]],
        y: [Double],
        indices: [Int],
        depth: Int,
        maxDepth: Int,
        minSamplesSplit: Int,
        maxFeatures: Int?,
        nodes: inout [FlatTreeNode]
    ) -> Int {
        let values = indices.map { y[$0] }
        let mean = values.mean()

        if depth >= maxDepth || indices.count < minSamplesSplit {
            nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: mean, isLeaf: true, impurityGain: 0.0))
            return nodes.count - 1
        }

        guard let split = bestSplit(X: X, y: y, indices: indices, criterion: .mse, maxFeatures: maxFeatures),
              split.gain > 0 else {
            nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: mean, isLeaf: true, impurityGain: 0.0))
            return nodes.count - 1
        }

        let currentIndex = nodes.count
        nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: 0, isLeaf: false, impurityGain: 0.0))

        let left  = buildTreeSync(X: X, y: y, indices: split.leftIndices,  depth: depth + 1, maxDepth: maxDepth, minSamplesSplit: minSamplesSplit, maxFeatures: maxFeatures, nodes: &nodes)
        let right = buildTreeSync(X: X, y: y, indices: split.rightIndices, depth: depth + 1, maxDepth: maxDepth, minSamplesSplit: minSamplesSplit, maxFeatures: maxFeatures, nodes: &nodes)

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
            if x[node.featureIndex] <= node.threshold {
                curr = node.leftChild
            } else {
                curr = node.rightChild
            }
        }
        return nodes[curr].value
    }
}
