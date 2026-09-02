import Foundation

// MARK: - GBDT Loss Functions

/// Loss functions available for Gradient Boosted Decision Tree optimization.
public enum GBDTLoss: Sendable, Equatable {
    /// Squared error loss ($L_2$) for conditional mean regression.
    case squaredError
    /// Absolute error loss ($L_1$) for conditional median regression.
    case absoluteError
    /// Pinball loss for conditional quantile regression at quantile $\alpha \in (0, 1)$.
    case quantile(alpha: Double)
}

// MARK: - Gradient Boosted Decision Trees (Regressor)

/// A native Swift GBDT Regressor using Flat Arrays (DOD).
/// Sequentially trains shallow regression trees on the residuals (pseudo-gradients) of the previous ensemble.
/// Supports mean squared error, absolute median error, and pinball quantile loss.
public actor GradientBoostedTreesRegressor: RegressorEstimator {
    /// The number of boosting estimators.
    public let nEstimators: Int
    /// The shrinkage learning rate.
    public let learningRate: Double
    /// The maximum tree depth.
    public let maxDepth: Int
    /// The minimum samples required to split an internal node.
    public let minSamplesSplit: Int
    /// The loss function utilized during boosting.
    public let loss: GBDTLoss
    
    // Forest stored as an array of flat tree node arrays (Data-Oriented Design)
    private var trees: [[FlatTreeNode]] = []
    private var initialPrediction: Double = 0.0
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - nEstimators: The number of boosting iterations.
    ///   - learningRate: The shrinkage parameter applied to each tree update.
    ///   - maxDepth: The maximum depth of individual regression trees.
    ///   - minSamplesSplit: The minimum number of samples to split a node.
    ///   - loss: The loss function to minimize (`.squaredError`, `.absoluteError`, or `.quantile(alpha:)`).
    /// - Throws: An error if hyperparameter values are invalid.
    public init(
        nEstimators: Int = 100,
        learningRate: Double = 0.1,
        maxDepth: Int = 3,
        minSamplesSplit: Int = 2,
        loss: GBDTLoss = .squaredError
    ) throws {
        guard nEstimators > 0 else { throw SwiftMLError.invalidParameter("nEstimators must be > 0") }
        guard learningRate > 0 else { throw SwiftMLError.invalidParameter("learningRate must be > 0") }
        if case .quantile(let alpha) = loss {
            guard alpha > 0.0 && alpha < 1.0 else {
                throw SwiftMLError.invalidParameter("quantile alpha must be between 0 and 1 (exclusive)")
            }
        }
        self.nEstimators = nEstimators
        self.learningRate = learningRate
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.loss = loss
    }
    
    /// Fits the GBDT model on the provided features and targets.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty else { throw SwiftMLError.emptyInput }
        guard features.count == targets.count else {
            throw SwiftMLError.dimensionMismatch(expected: features.count, got: targets.count)
        }
        
        let n = features.count
        switch loss {
        case .squaredError:
            initialPrediction = targets.mean()
        case .absoluteError:
            let sorted = targets.sorted()
            initialPrediction = sorted[sorted.count / 2]
        case .quantile(let alpha):
            let sorted = targets.sorted()
            let idx = Int((Double(sorted.count - 1) * alpha).rounded())
            initialPrediction = sorted[idx]
        }
        
        var currentPredictions = [Double](repeating: initialPrediction, count: n)
        var trainedTrees = [[FlatTreeNode]]()
        trainedTrees.reserveCapacity(nEstimators)
        
        let presorted = createPresortedIndices(X: features)
        
        for _ in 0..<nEstimators {
            let rawResiduals = zip(targets, currentPredictions).map { $0 - $1 }
            let pseudoResiduals: [Double]

            switch loss {
            case .squaredError:
                pseudoResiduals = rawResiduals
            case .absoluteError:
                pseudoResiduals = rawResiduals.map { $0 >= 0 ? 1.0 : -1.0 }
            case .quantile(let alpha):
                pseudoResiduals = rawResiduals.map { $0 >= 0 ? alpha : (alpha - 1.0) }
            }
            
            var nodes = [FlatTreeNode]()
            _ = GradientBoostedTreesRegressor.buildTree(
                X: features,
                y: pseudoResiduals,
                indices: Array(0..<n),
                presortedIndices: presorted,
                depth: 0,
                maxDepth: maxDepth,
                minSamplesSplit: minSamplesSplit,
                nodes: &nodes
            )

            if case .quantile(let alpha) = loss {
                GradientBoostedTreesRegressor.updateLeafQuantileValues(
                    nodes: &nodes,
                    features: features,
                    residuals: rawResiduals,
                    alpha: alpha
                )
            }

            trainedTrees.append(nodes)
            
            let lr = learningRate
            for i in 0..<n {
                let treePred = GradientBoostedTreesRegressor.predictSample(features[i], nodes: nodes)
                currentPredictions[i] += lr * treePred
            }
        }
        
        self.trees = trainedTrees
    }
    
    /// Returns predictions for the given feature matrix.
    public func predict(features: [[Double]]) async throws -> [Double] {
        guard !trees.isEmpty else { throw SwiftMLError.notFitted }
        
        let lr = learningRate
        let base = initialPrediction
        let count = features.count
        
        if count >= 1000 {
            var results = [Double](repeating: base, count: count)
            let localTrees = trees
            results.withUnsafeMutableBufferPointer { buf in
                guard let basePtr = buf.baseAddress else { return }
                struct UnsafeSendablePtr: @unchecked Sendable {
                    let ptr: UnsafeMutablePointer<Double>
                }
                let sendablePtr = UnsafeSendablePtr(ptr: basePtr)
                DispatchQueue.concurrentPerform(iterations: count) { i in
                    let sample = features[i]
                    var pred = base
                    for treeNodes in localTrees {
                        pred += lr * GradientBoostedTreesRegressor.predictSample(sample, nodes: treeNodes)
                    }
                    sendablePtr.ptr[i] = pred
                }
            }
            return results
        } else {
            return features.map { sample in
                var pred = base
                for treeNodes in trees {
                    pred += lr * GradientBoostedTreesRegressor.predictSample(sample, nodes: treeNodes)
                }
                return pred
            }
        }
    }
    
    // MARK: Private Helpers
    
    private static func updateLeafQuantileValues(
        nodes: inout [FlatTreeNode],
        features: [[Double]],
        residuals: [Double],
        alpha: Double
    ) {
        guard !nodes.isEmpty else { return }
        var leafResiduals: [Int: [Double]] = [:]
        for (i, x) in features.enumerated() {
            var curr = 0
            while !nodes[curr].isLeaf {
                let node = nodes[curr]
                if x[node.featureIndex] <= node.threshold {
                    curr = node.leftChild
                } else {
                    curr = node.rightChild
                }
            }
            leafResiduals[curr, default: []].append(residuals[i])
        }

        for (leafIdx, resList) in leafResiduals {
            guard !resList.isEmpty else { continue }
            let sorted = resList.sorted()
            let qIdx = Int((Double(sorted.count - 1) * alpha).rounded())
            let qVal = sorted[qIdx]
            let old = nodes[leafIdx]
            nodes[leafIdx] = FlatTreeNode(
                featureIndex: old.featureIndex,
                threshold: old.threshold,
                leftChild: old.leftChild,
                rightChild: old.rightChild,
                value: qVal,
                isLeaf: true
            )
        }
    }

    private static func buildTree(
        X: [[Double]],
        y: [Double],
        indices: [Int],
        presortedIndices: [[Int]],
        depth: Int,
        maxDepth: Int,
        minSamplesSplit: Int,
        nodes: inout [FlatTreeNode]
    ) -> Int {
        let values = indices.map { y[$0] }
        let mean = values.mean()
        
        if depth >= maxDepth || indices.count < minSamplesSplit {
            let leaf = FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: mean, isLeaf: true)
            nodes.append(leaf)
            return nodes.count - 1
        }
        
        guard let split = bestSplit(X: X, y: y, indices: indices, presortedIndices: presortedIndices, criterion: .mse, maxFeatures: nil) else {
            let leaf = FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: mean, isLeaf: true)
            nodes.append(leaf)
            return nodes.count - 1
        }
        
        let currentIndex = nodes.count
        nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: 0, isLeaf: false))
        
        let left = buildTree(X: X, y: y, indices: split.leftIndices, presortedIndices: presortedIndices, depth: depth + 1,
                             maxDepth: maxDepth, minSamplesSplit: minSamplesSplit, nodes: &nodes)
        let right = buildTree(X: X, y: y, indices: split.rightIndices, presortedIndices: presortedIndices, depth: depth + 1,
                              maxDepth: maxDepth, minSamplesSplit: minSamplesSplit, nodes: &nodes)
        
        nodes[currentIndex] = FlatTreeNode(featureIndex: split.featureIndex, threshold: split.threshold, leftChild: left, rightChild: right, value: mean, isLeaf: false)
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
