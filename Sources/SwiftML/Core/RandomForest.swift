import Foundation

// MARK: - Bootstrap Helper

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

    private var trees: [DecisionTreeNode] = []
    private var numClasses: Int = 0

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
        let maxDepth = self.maxDepth
        let maxFeatures = self.maxFeatures
        let minSamplesSplit = self.minSamplesSplit
        let criterion = self.criterion

        let trainedTrees: [DecisionTreeNode] = try await withThrowingTaskGroup(of: DecisionTreeNode.self) { group in
            for i in 0..<nEstimators {
                group.addTask {
                    let (bX, bY) = bootstrapSample(features: features, targets: targets, seed: i)
                    return RandomForestClassifier.buildTreeSync(
                        X: bX, y: bY,
                        indices: Array(0..<bX.count),
                        depth: 0,
                        maxDepth: maxDepth,
                        minSamplesSplit: minSamplesSplit,
                        criterion: criterion,
                        maxFeatures: maxFeatures
                    )
                }
            }
            var result = [DecisionTreeNode]()
            for try await tree in group {
                result.append(tree)
            }
            return result
        }

        self.trees = trainedTrees
    }

    public func predict(features: [[Double]]) throws -> [Int] {
        guard !trees.isEmpty else { throw MLError.notFitted }
        return features.map { sample in
            var votes = [Int: Int]()
            for tree in trees {
                let pred = RandomForestClassifier.predictSample(sample, node: tree)
                votes[Int(pred), default: 0] += 1
            }
            return votes.max(by: { $0.value < $1.value })?.key ?? 0
        }
    }

    public func predictProbability(features: [[Double]]) throws -> [[Double]] {
        guard !trees.isEmpty else { throw MLError.notFitted }
        return features.map { sample in
            var votes = [Double](repeating: 0, count: numClasses)
            for tree in trees {
                let pred = Int(RandomForestClassifier.predictSample(sample, node: tree))
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
        maxFeatures: Int?
    ) -> DecisionTreeNode {
        let labels = indices.map { y[$0] }

        if depth >= maxDepth || indices.count < minSamplesSplit || Set(labels).count == 1 {
            return DecisionTreeNode(value: labels.mostFrequent())
        }

        guard let split = bestSplit(X: X, y: y, indices: indices, criterion: criterion, maxFeatures: maxFeatures) else {
            return DecisionTreeNode(value: labels.mostFrequent())
        }

        let left = buildTreeSync(X: X, y: y, indices: split.leftIndices, depth: depth + 1,
                                  maxDepth: maxDepth, minSamplesSplit: minSamplesSplit,
                                  criterion: criterion, maxFeatures: maxFeatures)
        let right = buildTreeSync(X: X, y: y, indices: split.rightIndices, depth: depth + 1,
                                   maxDepth: maxDepth, minSamplesSplit: minSamplesSplit,
                                   criterion: criterion, maxFeatures: maxFeatures)
        return DecisionTreeNode(featureIndex: split.featureIndex, threshold: split.threshold, left: left, right: right)
    }

    private static func predictSample(_ x: [Double], node: DecisionTreeNode) -> Double {
        if node.isLeaf { return node.value ?? 0 }
        let fi = node.featureIndex!
        if x[fi] <= node.threshold! {
            return predictSample(x, node: node.left!)
        } else {
            return predictSample(x, node: node.right!)
        }
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

    private var trees: [DecisionTreeNode] = []

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

        let maxDepth = self.maxDepth
        let maxFeatures = self.maxFeatures
        let minSamplesSplit = self.minSamplesSplit

        let trainedTrees: [DecisionTreeNode] = try await withThrowingTaskGroup(of: DecisionTreeNode.self) { group in
            for i in 0..<nEstimators {
                group.addTask {
                    let (bX, bY) = bootstrapSample(features: features, targets: targets, seed: i)
                    return RandomForestRegressor.buildTreeSync(
                        X: bX, y: bY,
                        indices: Array(0..<bX.count),
                        depth: 0,
                        maxDepth: maxDepth,
                        minSamplesSplit: minSamplesSplit,
                        maxFeatures: maxFeatures
                    )
                }
            }
            var result = [DecisionTreeNode]()
            for try await tree in group {
                result.append(tree)
            }
            return result
        }

        self.trees = trainedTrees
    }

    public func predict(features: [[Double]]) throws -> [Double] {
        guard !trees.isEmpty else { throw MLError.notFitted }
        return features.map { sample in
            let preds = trees.map { RandomForestRegressor.predictSample(sample, node: $0) }
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
        maxFeatures: Int?
    ) -> DecisionTreeNode {
        let values = indices.map { y[$0] }
        let mean = values.mean()

        if depth >= maxDepth || indices.count < minSamplesSplit {
            return DecisionTreeNode(value: mean)
        }

        guard let split = bestSplit(X: X, y: y, indices: indices, criterion: .mse, maxFeatures: maxFeatures),
              split.gain > 0 else {
            return DecisionTreeNode(value: mean)
        }

        let left = buildTreeSync(X: X, y: y, indices: split.leftIndices, depth: depth + 1,
                                  maxDepth: maxDepth, minSamplesSplit: minSamplesSplit, maxFeatures: maxFeatures)
        let right = buildTreeSync(X: X, y: y, indices: split.rightIndices, depth: depth + 1,
                                   maxDepth: maxDepth, minSamplesSplit: minSamplesSplit, maxFeatures: maxFeatures)
        return DecisionTreeNode(featureIndex: split.featureIndex, threshold: split.threshold, left: left, right: right)
    }

    private static func predictSample(_ x: [Double], node: DecisionTreeNode) -> Double {
        if node.isLeaf { return node.value ?? 0 }
        let fi = node.featureIndex!
        if x[fi] <= node.threshold! {
            return predictSample(x, node: node.left!)
        } else {
            return predictSample(x, node: node.right!)
        }
    }
}
