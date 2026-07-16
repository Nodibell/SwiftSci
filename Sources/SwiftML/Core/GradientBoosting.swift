import Foundation

// MARK: - Gradient Boosted Decision Trees (Regressor)

/// A native Swift GBDT Regressor.
/// Sequentially trains shallow regression trees on the residuals (pseudo-gradients) of the previous ensemble.
public actor GradientBoostedTreesRegressor: RegressorEstimator {
    public let nEstimators: Int
    public let learningRate: Double
    public let maxDepth: Int
    public let minSamplesSplit: Int

    private var trees: [DecisionTreeNode] = []
    private var initialPrediction: Double = 0.0

    public init(
        nEstimators: Int = 100,
        learningRate: Double = 0.1,
        maxDepth: Int = 3,
        minSamplesSplit: Int = 2
    ) throws {
        guard nEstimators > 0 else { throw MLError.invalidParameter("nEstimators must be > 0") }
        guard learningRate > 0 else { throw MLError.invalidParameter("learningRate must be > 0") }
        self.nEstimators = nEstimators
        self.learningRate = learningRate
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
    }

    /// Fits the GBDT model on the provided features and targets.
    public func fit(features: [[Double]], targets: [Double]) throws {
        guard !features.isEmpty else { throw MLError.emptyInput }
        guard features.count == targets.count else {
            throw MLError.dimensionMismatch(expected: features.count, got: targets.count)
        }

        let n = features.count

        // Start with the mean of all targets as the first (constant) prediction
        initialPrediction = targets.mean()

        var currentPredictions = [Double](repeating: initialPrediction, count: n)
        var trainedTrees = [DecisionTreeNode]()
        trainedTrees.reserveCapacity(nEstimators)

        for _ in 0..<nEstimators {
            // Compute pseudo-residuals (gradient = y - prediction for MSE loss)
            let residuals = zip(targets, currentPredictions).map { $0 - $1 }

            // Train a shallow regression tree on residuals
            let tree = GradientBoostedTreesRegressor.buildTree(
                X: features,
                y: residuals,
                indices: Array(0..<n),
                depth: 0,
                maxDepth: maxDepth,
                minSamplesSplit: minSamplesSplit
            )
            trainedTrees.append(tree)

            // Update predictions
            let lr = learningRate
            for i in 0..<n {
                let treePred = GradientBoostedTreesRegressor.predictSample(features[i], node: tree)
                currentPredictions[i] += lr * treePred
            }
        }

        self.trees = trainedTrees
    }

    /// Returns predictions for the given feature matrix.
    public func predict(features: [[Double]]) throws -> [Double] {
        guard !trees.isEmpty else { throw MLError.notFitted }
        let lr = learningRate
        let base = initialPrediction
        return features.map { sample in
            var pred = base
            for tree in trees {
                pred += lr * GradientBoostedTreesRegressor.predictSample(sample, node: tree)
            }
            return pred
        }
    }

    // MARK: Private Helpers

    private static func buildTree(
        X: [[Double]],
        y: [Double],
        indices: [Int],
        depth: Int,
        maxDepth: Int,
        minSamplesSplit: Int
    ) -> DecisionTreeNode {
        let values = indices.map { y[$0] }
        let mean = values.mean()

        if depth >= maxDepth || indices.count < minSamplesSplit {
            return DecisionTreeNode(value: mean)
        }

        guard let split = bestSplit(X: X, y: y, indices: indices, criterion: .mse, maxFeatures: nil) else {
            return DecisionTreeNode(value: mean)
        }

        let left = buildTree(X: X, y: y, indices: split.leftIndices, depth: depth + 1,
                              maxDepth: maxDepth, minSamplesSplit: minSamplesSplit)
        let right = buildTree(X: X, y: y, indices: split.rightIndices, depth: depth + 1,
                               maxDepth: maxDepth, minSamplesSplit: minSamplesSplit)
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
