import Foundation

// MARK: - Split Criteria

/// The impurity criterion used to evaluate candidate splits in a decision tree.
///
/// - ``gini``: Gini impurity — minimizes class mixing probability.
/// - ``entropy``: Information gain — minimizes Shannon entropy.
/// - ``mse``: Mean Squared Error — used for regression trees.
public enum SplitCriterion: Sendable {
    /// Gini impurity criterion for classification trees.
    case gini
    /// Entropy / information gain criterion for classification trees.
    case entropy
    /// Mean Squared Error criterion for regression trees.
    case mse
}

// MARK: - Internal Types

struct SplitResult: Sendable {
    let featureIndex: Int
    let threshold: Double
    let gain: Double
    let leftIndices: [Int]
    let rightIndices: [Int]
}

// MARK: - Flat Decision Tree Node (DOD Architecture)

/// A flat, cache-friendly representation of a decision tree node.
///
/// Stores all tree structure in a contiguous array rather than pointer-linked objects,
/// eliminating ARC overhead and maximizing CPU cache locality (Data-Oriented Design).
public struct FlatTreeNode: Sendable, Codable {
    /// Index of the feature to split on. `-1` for leaf nodes.
    public let featureIndex: Int
    /// Threshold value for the binary split condition (`feature[i] <= threshold`).
    public let threshold: Double
    /// Index of the left child node in the flat tree array (`feature <= threshold`).
    public let leftChild: Int
    /// Index of the right child node in the flat tree array (`feature > threshold`).
    public let rightChild: Int
    /// Leaf value: predicted class label (classifier) or predicted mean (regressor).
    public let value: Double
    /// `true` if this node is a leaf with no further splits.
    public let isLeaf: Bool
    /// Impurity reduction achieved by this split.
    public let impurityGain: Double

    /// Creates a new flat tree node.
    /// - Parameters:
    ///   - featureIndex: Index of the feature used to split. Use `-1` for leaf nodes.
    ///   - threshold: Split threshold value.
    ///   - leftChild: Index of the left child node in the flat tree array.
    ///   - rightChild: Index of the right child node in the flat tree array.
    ///   - value: Predicted value at a leaf node.
    ///   - isLeaf: Whether this node is a terminal leaf.
    ///   - impurityGain: Impurity reduction achieved by this split.
    public init(
        featureIndex: Int,
        threshold: Double,
        leftChild: Int,
        rightChild: Int,
        value: Double,
        isLeaf: Bool,
        impurityGain: Double = 0.0
    ) {
        self.featureIndex = featureIndex
        self.threshold = threshold
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.value = value
        self.isLeaf = isLeaf
        self.impurityGain = impurityGain
    }
}

// MARK: - Shared helpers

func giniImpurity(_ labels: [Double]) -> Double {
    guard !labels.isEmpty else { return 0 }
    var counts = [Double: Int]()
    for l in labels { counts[l, default: 0] += 1 }
    let n = Double(labels.count)
    return 1.0 - counts.values.reduce(0.0) { $0 + pow(Double($1) / n, 2) }
}

func entropy(_ labels: [Double]) -> Double {
    guard !labels.isEmpty else { return 0 }
    var counts = [Double: Int]()
    for l in labels { counts[l, default: 0] += 1 }
    let n = Double(labels.count)
    return -counts.values.reduce(0.0) { acc, count in
        let p = Double(count) / n
        return acc + (p > 0 ? p * log2(p) : 0)
    }
}

import Accelerate

func mseImpurity(_ values: [Double]) -> Double {
    guard values.count > 1 else { return 0.0 }
    let meanVal = vDSP.mean(values)
    let sqMean = vDSP.meanSquare(values)
    return max(0.0, sqMean - (meanVal * meanVal))
}

// MARK: - Pre-sorted Feature Matrix Helper

func createPresortedIndices(X: [[Double]]) -> [[Int]] {
    guard !X.isEmpty else { return [] }
    let numFeatures = X[0].count
    let numSamples = X.count
    var presorted = [[Int]]()
    presorted.reserveCapacity(numFeatures)
    for f in 0..<numFeatures {
        let sortedF = (0..<numSamples).sorted { X[$0][f] < X[$1][f] }
        presorted.append(sortedF)
    }
    return presorted
}

func bestSplit(
    X: [[Double]],
    y: [Double],
    indices: [Int],
    presortedIndices: [[Int]]? = nil,
    criterion: SplitCriterion,
    maxFeatures: Int?
) -> SplitResult? {
    guard indices.count > 1 else { return nil }

    let numFeatures = X[0].count
    let featureRange: [Int]
    if let maxF = maxFeatures, maxF < numFeatures {
        featureRange = Array((0..<numFeatures).shuffled().prefix(maxF))
    } else {
        featureRange = Array(0..<numFeatures)
    }

    var best: SplitResult? = nil
    let n = Double(indices.count)

    let inNodeMask: [Bool]?
    if presortedIndices != nil {
        var mask = [Bool](repeating: false, count: X.count)
        for idx in indices { mask[idx] = true }
        inNodeMask = mask
    } else {
        inNodeMask = nil
    }

    if criterion == .mse {
        var totalSum = 0.0
        for idx in indices { totalSum += y[idx] }
        let baseTerm = (totalSum * totalSum) / n

        for fi in featureRange {
            let sortedIndices: [Int]
            if let presorted = presortedIndices, let mask = inNodeMask {
                sortedIndices = presorted[fi].filter { mask[$0] }
            } else {
                sortedIndices = indices.sorted { X[$0][fi] < X[$1][fi] }
            }

            var leftSum = 0.0
            var leftCount = 0.0

            for i in 0..<(sortedIndices.count - 1) {
                let idx = sortedIndices[i]
                leftSum += y[idx]
                leftCount += 1.0

                let nextIdx = sortedIndices[i + 1]
                
                if X[idx][fi] < X[nextIdx][fi] {
                    let rightSum = totalSum - leftSum
                    let rightCount = n - leftCount

                    let gain = ( (leftSum * leftSum / leftCount) + (rightSum * rightSum / rightCount) - baseTerm ) / n

                    if best == nil || gain > (best?.gain ?? -Double.infinity) {
                        let threshold = (X[idx][fi] + X[nextIdx][fi]) / 2.0
                        best = SplitResult(
                            featureIndex: fi,
                            threshold: threshold,
                            gain: gain,
                            leftIndices: Array(sortedIndices[0...i]),
                            rightIndices: Array(sortedIndices[(i+1)...])
                        )
                    }
                }
            }
        }
    } else {
        var totalCounts = [Double: Int]()
        for idx in indices { totalCounts[y[idx], default: 0] += 1 }

        let parentImpurity: Double
        if criterion == .gini {
            let sumSq = totalCounts.values.reduce(0.0) { $0 + Double($1 * $1) }
            parentImpurity = 1.0 - (sumSq / (n * n))
        } else {
            parentImpurity = totalCounts.values.reduce(0.0) { acc, count in
                let p = Double(count) / n
                return acc - (p > 0 ? p * log2(p) : 0)
            }
        }

        for fi in featureRange {
            let sortedIndices: [Int]
            if let presorted = presortedIndices, let mask = inNodeMask {
                sortedIndices = presorted[fi].filter { mask[$0] }
            } else {
                sortedIndices = indices.sorted { X[$0][fi] < X[$1][fi] }
            }

            var leftCounts = [Double: Int]()
            var rightCounts = totalCounts
            var leftCount = 0.0

            for i in 0..<(sortedIndices.count - 1) {
                let idx = sortedIndices[i]
                let target = y[idx]

                // Оновлюємо частоти за O(1)
                leftCounts[target, default: 0] += 1
                rightCounts[target, default: 0] -= 1
                leftCount += 1.0

                let nextIdx = sortedIndices[i + 1]
                if X[idx][fi] < X[nextIdx][fi] {
                    let rightCount = n - leftCount
                    let leftImpurity: Double
                    let rightImpurity: Double

                    if criterion == .gini {
                        let leftSumSq = leftCounts.values.reduce(0.0) { $0 + Double($1 * $1) }
                        leftImpurity = 1.0 - (leftSumSq / (leftCount * leftCount))

                        let rightSumSq = rightCounts.values.reduce(0.0) { $0 + Double($1 * $1) }
                        rightImpurity = 1.0 - (rightSumSq / (rightCount * rightCount))
                    } else {
                        leftImpurity = leftCounts.values.reduce(0.0) { acc, count in
                            let p = Double(count) / leftCount
                            return acc - (p > 0 ? p * log2(p) : 0)
                        }
                        rightImpurity = rightCounts.values.reduce(0.0) { acc, count in
                            let p = Double(count) / rightCount
                            return acc - (p > 0 ? p * log2(p) : 0)
                        }
                    }

                    let gain = parentImpurity - (leftCount / n) * leftImpurity - (rightCount / n) * rightImpurity

                    if best == nil || gain > (best?.gain ?? -Double.infinity) {
                        let threshold = (X[idx][fi] + X[nextIdx][fi]) / 2.0
                        best = SplitResult(
                            featureIndex: fi,
                            threshold: threshold,
                            gain: gain,
                            leftIndices: Array(sortedIndices[0...i]),
                            rightIndices: Array(sortedIndices[(i+1)...])
                        )
                    }
                }
            }
        }
    }

    return best
}
func computeFeatureImportances(nodes: [FlatTreeNode], numFeatures: Int) -> [Double]? {
    guard numFeatures > 0, !nodes.isEmpty else { return nil }
    var importances = [Double](repeating: 0.0, count: numFeatures)
    for node in nodes where !node.isLeaf && node.featureIndex >= 0 && node.featureIndex < numFeatures {
        importances[node.featureIndex] += node.impurityGain
    }
    let totalGain = importances.reduce(0.0, +)
    if totalGain > 0 {
        return importances.map { $0 / totalGain }
    }
    return importances
}

// MARK: - Decision Tree Classifier

/// A pure Swift Decision Tree Classifier using Gini or Entropy splitting.
public actor DecisionTreeClassifier: ClassifierEstimator {
    /// The max depth.
    public let maxDepth: Int
    /// The min samples split.
    public let minSamplesSplit: Int
    /// The criterion.
    public let criterion: SplitCriterion

    private var nodes: [FlatTreeNode] = []
    
    /// The collection of fitted tree nodes in contiguous flat array.
    public var flatNodes: [FlatTreeNode] { nodes }
    
    private var numFeatures: Int = 0

    /// The feature importances.
    public var featureImportances: [Double]? {
        computeFeatureImportances(nodes: nodes, numFeatures: numFeatures)
    }

    /// Creates a new instance.
    /// - Parameters:
    ///   - maxDepth: The max depth.
    ///   - minSamplesSplit: The min samples split.
    ///   - criterion: The criterion.
    public init(maxDepth: Int = 5, minSamplesSplit: Int = 2, criterion: SplitCriterion = .gini) {
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.criterion = criterion
    }

    /// Fit.
    /// - Parameters:
    ///   - features: The features.
    ///   - targets: The targets.
    /// - Throws: An error if the operation fails.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty else { throw SwiftMLError.emptyInput }
        guard features.count == targets.count else {
            throw SwiftMLError.dimensionMismatch(expected: features.count, got: targets.count)
        }
        numFeatures = features[0].count
        nodes = []
        let presorted = createPresortedIndices(X: features)
        _ = buildTree(X: features, y: targets, indices: Array(0..<features.count), presortedIndices: presorted, depth: 0, nodes: &nodes)
    }

    /// Predict.
    /// - Parameters:
    ///   - features: The features.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[Int]` result.
    public func predict(features: [[Double]]) async throws -> [Int] {
        guard !nodes.isEmpty else { throw SwiftMLError.notFitted }
        return features.map { predictSample($0, nodes: nodes) }
    }

    /// Predict probability.
    /// - Parameters:
    ///   - features: The features.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[[Double]]` result.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        guard !nodes.isEmpty else { throw SwiftMLError.notFitted }
        let maxLabel = nodes.map { Int($0.value) }.max() ?? 0
        let numClasses = maxLabel + 1
        return features.map { sample in
            let predClass = predictSample(sample, nodes: nodes)
            var probs = [Double](repeating: 0.0, count: numClasses)
            if predClass < probs.count {
                probs[predClass] = 1.0
            }
            return probs
        }
    }
    
    /// Get tree nodes.
    /// - Returns: A `[FlatTreeNode]` result.
    public func getTreeNodes() -> [FlatTreeNode] {
        return nodes
    }

    /// Prunes the decision tree using Minimal Cost-Complexity Pruning with threshold parameter `alpha`.
    /// - Parameter alpha: Complexity parameter used for pruning. Subtrees with impurity gain <= alpha are collapsed into leaf nodes.
    public func prune(alpha: Double) {
        guard alpha > 0, !nodes.isEmpty else { return }
        func pruneNode(_ idx: Int) -> Bool {
            guard idx >= 0 && idx < nodes.count else { return true }
            if nodes[idx].isLeaf { return true }

            let leftLeaf = pruneNode(nodes[idx].leftChild)
            let rightLeaf = pruneNode(nodes[idx].rightChild)

            if leftLeaf && rightLeaf && nodes[idx].impurityGain <= alpha {
                nodes[idx] = FlatTreeNode(
                    featureIndex: -1,
                    threshold: 0.0,
                    leftChild: -1,
                    rightChild: -1,
                    value: nodes[idx].value,
                    isLeaf: true,
                    impurityGain: 0.0
                )
                return true
            }
            return false
        }
        _ = pruneNode(0)
    }

    private func buildTree(X: [[Double]], y: [Double], indices: [Int], presortedIndices: [[Int]], depth: Int, nodes: inout [FlatTreeNode]) -> Int {
        let labels = indices.map { y[$0] }
        let majority = labels.mostFrequent()

        // Leaf: max depth, too few samples, or pure node
        if depth >= maxDepth || indices.count < minSamplesSplit || Set(labels).count == 1 {
            let leaf = FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: majority, isLeaf: true, impurityGain: 0.0)
            nodes.append(leaf)
            return nodes.count - 1
        }

        guard let split = bestSplit(X: X, y: y, indices: indices, presortedIndices: presortedIndices, criterion: criterion, maxFeatures: nil) else {
            let leaf = FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: majority, isLeaf: true, impurityGain: 0.0)
            nodes.append(leaf)
            return nodes.count - 1
        }

        let currentIndex = nodes.count
        // Placeholder to maintain index stability during recursion
        nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: 0, isLeaf: false, impurityGain: 0.0))

        let leftIndex  = buildTree(X: X, y: y, indices: split.leftIndices,  presortedIndices: presortedIndices, depth: depth + 1, nodes: &nodes)
        let rightIndex = buildTree(X: X, y: y, indices: split.rightIndices, presortedIndices: presortedIndices, depth: depth + 1, nodes: &nodes)
        
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

    private func predictSample(_ x: [Double], nodes: [FlatTreeNode]) -> Int {
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
        return Int(nodes[curr].value)
    }
}

// MARK: - Decision Tree Regressor

/// A pure Swift Decision Tree Regressor using MSE splitting.
public actor DecisionTreeRegressor: RegressorEstimator {
    /// The max depth.
    public let maxDepth: Int
    /// The min samples split.
    public let minSamplesSplit: Int

    private var nodes: [FlatTreeNode] = []
    
    /// The collection of fitted tree nodes in contiguous flat array.
    public var flatNodes: [FlatTreeNode] { nodes }
    
    private var numFeatures: Int = 0

    /// The feature importances.
    public var featureImportances: [Double]? {
        computeFeatureImportances(nodes: nodes, numFeatures: numFeatures)
    }

    /// Creates a new instance.
    /// - Parameters:
    ///   - maxDepth: The max depth.
    ///   - minSamplesSplit: The min samples split.
    public init(maxDepth: Int = 5, minSamplesSplit: Int = 2) {
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
    }

    /// Fit.
    /// - Parameters:
    ///   - features: The features.
    ///   - targets: The targets.
    /// - Throws: An error if the operation fails.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty else { throw SwiftMLError.emptyInput }
        guard features.count == targets.count else {
            throw SwiftMLError.dimensionMismatch(expected: features.count, got: targets.count)
        }
        
        numFeatures = features[0].count
        nodes = []
        let presorted = createPresortedIndices(X: features)
        _ = buildTree(X: features, y: targets, indices: Array(0..<features.count), presortedIndices: presorted, depth: 0, nodes: &nodes)
    }

    /// Predict.
    /// - Parameters:
    ///   - features: The features.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[Double]` result.
    public func predict(features: [[Double]]) async throws -> [Double] {
        guard !nodes.isEmpty else { throw SwiftMLError.notFitted }
        return features.map { predictSample($0, nodes: nodes) }
    }

    /// Prunes the regression tree using Minimal Cost-Complexity Pruning with threshold parameter `alpha`.
    /// - Parameter alpha: Complexity parameter used for pruning. Subtrees with impurity gain <= alpha are collapsed into leaf nodes.
    public func prune(alpha: Double) {
        guard alpha > 0, !nodes.isEmpty else { return }
        func pruneNode(_ idx: Int) -> Bool {
            guard idx >= 0 && idx < nodes.count else { return true }
            if nodes[idx].isLeaf { return true }

            let leftLeaf = pruneNode(nodes[idx].leftChild)
            let rightLeaf = pruneNode(nodes[idx].rightChild)

            if leftLeaf && rightLeaf && nodes[idx].impurityGain <= alpha {
                nodes[idx] = FlatTreeNode(
                    featureIndex: -1,
                    threshold: 0.0,
                    leftChild: -1,
                    rightChild: -1,
                    value: nodes[idx].value,
                    isLeaf: true,
                    impurityGain: 0.0
                )
                return true
            }
            return false
        }
        _ = pruneNode(0)
    }

    /// Get tree nodes.
    /// - Returns: A `[FlatTreeNode]` result.
    public func getTreeNodes() -> [FlatTreeNode] {
        return nodes
    }

    private func buildTree(X: [[Double]], y: [Double], indices: [Int], presortedIndices: [[Int]], depth: Int, nodes: inout [FlatTreeNode]) -> Int {
        let values = indices.map { y[$0] }
        let mean = values.mean()

        if depth >= maxDepth || indices.count < minSamplesSplit {
            let leaf = FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: mean, isLeaf: true, impurityGain: 0.0)
            nodes.append(leaf)
            return nodes.count - 1
        }

        guard let split = bestSplit(X: X, y: y, indices: indices, presortedIndices: presortedIndices, criterion: .mse, maxFeatures: nil) else {
            let leaf = FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: mean, isLeaf: true, impurityGain: 0.0)
            nodes.append(leaf)
            return nodes.count - 1
        }

        let currentIndex = nodes.count
        // Placeholder
        nodes.append(FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: 0, isLeaf: false, impurityGain: 0.0))

        let leftIndex  = buildTree(X: X, y: y, indices: split.leftIndices,  presortedIndices: presortedIndices, depth: depth + 1, nodes: &nodes)
        let rightIndex = buildTree(X: X, y: y, indices: split.rightIndices, presortedIndices: presortedIndices, depth: depth + 1, nodes: &nodes)
        
        let nodeGain = split.gain * Double(indices.count)
        nodes[currentIndex] = FlatTreeNode(
            featureIndex: split.featureIndex,
            threshold: split.threshold,
            leftChild: leftIndex,
            rightChild: rightIndex,
            value: mean,
            isLeaf: false,
            impurityGain: nodeGain
        )

        return currentIndex
    }

    private func predictSample(_ x: [Double], nodes: [FlatTreeNode]) -> Double {
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

// MARK: - Array Extension

extension Array where Element == Double {
    func mostFrequent() -> Double {
        var counts = [Double: Int]()
        for v in self { counts[v, default: 0] += 1 }
        return counts.max(by: { $0.value < $1.value })?.key ?? 0
    }

    func mean() -> Double {
        guard !isEmpty else { return 0 }
        return reduce(0, +) / Double(count)
    }
}
