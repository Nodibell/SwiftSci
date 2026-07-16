import Foundation

// MARK: - Errors



// MARK: - Split Criteria

public enum SplitCriterion: Sendable {
    case gini
    case entropy
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

// MARK: - Decision Tree Node

public final class DecisionTreeNode: Sendable {
    public let featureIndex: Int?
    public let threshold: Double?
    public let left: DecisionTreeNode?
    public let right: DecisionTreeNode?
    /// Leaf value: predicted class (classifier) or mean (regressor)
    public let value: Double?

    public var isLeaf: Bool { left == nil && right == nil }

    init(featureIndex: Int, threshold: Double, left: DecisionTreeNode, right: DecisionTreeNode) {
        self.featureIndex = featureIndex
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = nil
    }

    init(value: Double) {
        self.featureIndex = nil
        self.threshold = nil
        self.left = nil
        self.right = nil
        self.value = value
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

func mseImpurity(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let mean = values.reduce(0, +) / Double(values.count)
    return values.reduce(0.0) { $0 + pow($1 - mean, 2) } / Double(values.count)
}

func bestSplit(
    X: [[Double]],
    y: [Double],
    indices: [Int],
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

    let impurityFn: ([Double]) -> Double
    switch criterion {
    case .gini:    impurityFn = giniImpurity
    case .entropy: impurityFn = entropy
    case .mse:     impurityFn = mseImpurity
    }

    let parentImpurity = impurityFn(indices.map { y[$0] })

    for fi in featureRange {
        // Collect unique sorted thresholds
        let values = indices.map { X[$0][fi] }.sorted()
        let thresholds = zip(values, values.dropFirst()).map { ($0 + $1) / 2 }
        let uniqueThresholds = Array(Set(thresholds)).sorted()

        for threshold in uniqueThresholds {
            let left  = indices.filter { X[$0][fi] <= threshold }
            let right = indices.filter { X[$0][fi] >  threshold }
            guard !left.isEmpty, !right.isEmpty else { continue }

            let n = Double(indices.count)
            let gain = parentImpurity
                - Double(left.count)  / n * impurityFn(left.map  { y[$0] })
                - Double(right.count) / n * impurityFn(right.map { y[$0] })

            if best == nil || gain > best!.gain {
                best = SplitResult(
                    featureIndex: fi,
                    threshold: threshold,
                    gain: gain,
                    leftIndices: left,
                    rightIndices: right
                )
            }
        }
    }
    return best
}

// MARK: - Decision Tree Classifier

/// A pure Swift Decision Tree Classifier using Gini or Entropy splitting.
public actor DecisionTreeClassifier: ClassifierEstimator {
    public let maxDepth: Int
    public let minSamplesSplit: Int
    public let criterion: SplitCriterion

    private var root: DecisionTreeNode?

    public init(maxDepth: Int = 5, minSamplesSplit: Int = 2, criterion: SplitCriterion = .gini) {
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.criterion = criterion
    }

    public func fit(features: [[Double]], targets: [Double]) throws {
        guard !features.isEmpty else { throw MLError.emptyInput }
        guard features.count == targets.count else {
            throw MLError.dimensionMismatch(expected: features.count, got: targets.count)
        }
        root = buildTree(X: features, y: targets, indices: Array(0..<features.count), depth: 0)
    }

    public func predict(features: [[Double]]) throws -> [Int] {
        guard let root = root else { throw MLError.notFitted }
        return features.map { predictSample($0, node: root) }
    }

    private func buildTree(X: [[Double]], y: [Double], indices: [Int], depth: Int) -> DecisionTreeNode {
        let labels = indices.map { y[$0] }

        // Leaf: max depth, too few samples, or pure node
        if depth >= maxDepth || indices.count < minSamplesSplit || Set(labels).count == 1 {
            let majority = labels.mostFrequent()
            return DecisionTreeNode(value: majority)
        }

        guard let split = bestSplit(X: X, y: y, indices: indices, criterion: criterion, maxFeatures: nil) else {
            return DecisionTreeNode(value: labels.mostFrequent())
        }

        let left  = buildTree(X: X, y: y, indices: split.leftIndices,  depth: depth + 1)
        let right = buildTree(X: X, y: y, indices: split.rightIndices, depth: depth + 1)
        return DecisionTreeNode(featureIndex: split.featureIndex, threshold: split.threshold, left: left, right: right)
    }

    private func predictSample(_ x: [Double], node: DecisionTreeNode) -> Int {
        if node.isLeaf { return Int(node.value ?? 0) }
        let fi = node.featureIndex!
        if x[fi] <= node.threshold! {
            return predictSample(x, node: node.left!)
        } else {
            return predictSample(x, node: node.right!)
        }
    }
}

// MARK: - Decision Tree Regressor

/// A pure Swift Decision Tree Regressor using MSE splitting.
public actor DecisionTreeRegressor: RegressorEstimator {
    public let maxDepth: Int
    public let minSamplesSplit: Int

    private var root: DecisionTreeNode?

    public init(maxDepth: Int = 5, minSamplesSplit: Int = 2) {
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
    }

    public func fit(features: [[Double]], targets: [Double]) throws {
        guard !features.isEmpty else { throw MLError.emptyInput }
        guard features.count == targets.count else {
            throw MLError.dimensionMismatch(expected: features.count, got: targets.count)
        }
        root = buildTree(X: features, y: targets, indices: Array(0..<features.count), depth: 0)
    }

    public func predict(features: [[Double]]) throws -> [Double] {
        guard let root = root else { throw MLError.notFitted }
        return features.map { predictSample($0, node: root) }
    }

    public func getRoot() -> DecisionTreeNode? {
        return root
    }

    private func buildTree(X: [[Double]], y: [Double], indices: [Int], depth: Int) -> DecisionTreeNode {
        let values = indices.map { y[$0] }
        let mean = values.reduce(0, +) / Double(values.count)

        if depth >= maxDepth || indices.count < minSamplesSplit {
            return DecisionTreeNode(value: mean)
        }

        guard let split = bestSplit(X: X, y: y, indices: indices, criterion: .mse, maxFeatures: nil) else {
            return DecisionTreeNode(value: mean)
        }

        let left  = buildTree(X: X, y: y, indices: split.leftIndices,  depth: depth + 1)
        let right = buildTree(X: X, y: y, indices: split.rightIndices, depth: depth + 1)
        return DecisionTreeNode(featureIndex: split.featureIndex, threshold: split.threshold, left: left, right: right)
    }

    private func predictSample(_ x: [Double], node: DecisionTreeNode) -> Double {
        if node.isLeaf { return node.value ?? 0 }
        let fi = node.featureIndex!
        if x[fi] <= node.threshold! {
            return predictSample(x, node: node.left!)
        } else {
            return predictSample(x, node: node.right!)
        }
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
