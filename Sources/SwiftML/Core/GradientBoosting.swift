import Foundation

// MARK: - Gradient Boosted Decision Trees (Regressor)

/// A native Swift GBDT Regressor using Flat Arrays (DOD).
/// Sequentially trains shallow regression trees on the residuals (pseudo-gradients) of the previous ensemble.
public actor GradientBoostedTreesRegressor: RegressorEstimator {
    /// The n estimators.
    public let nEstimators: Int
    /// The learning rate.
    public let learningRate: Double
    /// The max depth.
    public let maxDepth: Int
    /// The min samples split.
    public let minSamplesSplit: Int
    
    // Forest stored as an array of flat tree node arrays (Data-Oriented Design)
    private var trees: [[FlatTreeNode]] = []
    private var initialPrediction: Double = 0.0
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - nEstimators: The n estimators.
    ///   - learningRate: The learning rate.
    ///   - maxDepth: The max depth.
    ///   - minSamplesSplit: The min samples split.
    /// - Throws: An error if the operation fails.
    public init(
        nEstimators: Int = 100,
        learningRate: Double = 0.1,
        maxDepth: Int = 3,
        minSamplesSplit: Int = 2
    ) throws {
        guard nEstimators > 0 else { throw SwiftMLError.invalidParameter("nEstimators must be > 0") }
        guard learningRate > 0 else { throw SwiftMLError.invalidParameter("learningRate must be > 0") }
        self.nEstimators = nEstimators
        self.learningRate = learningRate
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
    }
    
    /// Fits the GBDT model on the provided features and targets.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty else { throw SwiftMLError.emptyInput }
        guard features.count == targets.count else {
            throw SwiftMLError.dimensionMismatch(expected: features.count, got: targets.count)
        }
        
        let n = features.count
        initialPrediction = targets.mean()
        
        var currentPredictions = [Double](repeating: initialPrediction, count: n)
        var trainedTrees = [[FlatTreeNode]]()
        trainedTrees.reserveCapacity(nEstimators)
        
        let presorted = createPresortedIndices(X: features)
        
        for _ in 0..<nEstimators {
            let residuals = zip(targets, currentPredictions).map { $0 - $1 }
            
            var nodes = [FlatTreeNode]()
            _ = GradientBoostedTreesRegressor.buildTree(
                X: features,
                y: residuals,
                indices: Array(0..<n),
                presortedIndices: presorted,
                depth: 0,
                maxDepth: maxDepth,
                minSamplesSplit: minSamplesSplit,
                nodes: &nodes
            )
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
