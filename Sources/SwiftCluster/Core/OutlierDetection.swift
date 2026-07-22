import Foundation
import SwiftPreprocessing
import SwiftStats

private struct SeededRandom {
    private var state: UInt64
    init(seed: Int) {
        self.state = UInt64(bitPattern: Int64(seed &+ 1))
    }
    mutating func nextInt(upperBound: Int) -> Int {
        guard upperBound > 0 else { return 0 }
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Int(state >> 33) % upperBound
    }
}

/// Anomaly score and binary classification output for outlier detection models.
public struct AnomalyPrediction: Sendable {
    /// Binary label: 1 for inliers, -1 for outliers.
    public let labels: [Int]
    /// Anomaly scores (higher score indicates higher likelihood of anomaly).
    public let scores: [Double]
}

/// Isolation Forest algorithm for anomaly detection using random partitions.
public final class IsolationForest: Sendable {
    public let nEstimators: Int
    public let maxSamples: Int?
    public let contamination: Double
    
    private final class Node: @unchecked Sendable {
        var featureIndex: Int = 0
        var splitValue: Double = 0.0
        var left: Node?
        var right: Node?
        var size: Int = 0
        var isLeaf: Bool { left == nil && right == nil }
    }
    
    private final class IsolationTree: @unchecked Sendable {
        let root: Node
        init(root: Node) { self.root = root }
    }
    
    private let trees: [IsolationTree]
    private let thresholdScore: Double
    
    public init(nEstimators: Int = 100, maxSamples: Int? = nil, contamination: Double = 0.1) {
        self.nEstimators = nEstimators
        self.maxSamples = maxSamples
        self.contamination = max(0.0, min(0.5, contamination))
        self.trees = []
        self.thresholdScore = 0.5
    }
    
    private init(trees: [IsolationTree], thresholdScore: Double, nEstimators: Int, maxSamples: Int?, contamination: Double) {
        self.trees = trees
        self.thresholdScore = thresholdScore
        self.nEstimators = nEstimators
        self.maxSamples = maxSamples
        self.contamination = contamination
    }
    
    /// Fits the Isolation Forest model on feature matrix.
    public static func fit(data: [[Double]], nEstimators: Int = 100, maxSamples: Int? = nil, contamination: Double = 0.1, seed: UInt64 = 42) throws -> IsolationForest {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        let numSamples = data.count
        let numFeatures = data[0].count
        let subsampleSize = min(maxSamples ?? min(256, numSamples), numSamples)
        let maxDepth = Int(ceil(log2(Double(max(2, subsampleSize)))))
        
        var rng = SeededRandom(seed: Int(seed))
        var builtTrees: [IsolationTree] = []
        
        for _ in 0..<nEstimators {
            let sampleIndices = (0..<subsampleSize).map { _ in rng.nextInt(upperBound: numSamples) }
            let subData = sampleIndices.map { data[$0] }
            let root = buildTree(data: subData, currentDepth: 0, maxDepth: maxDepth, numFeatures: numFeatures, rng: &rng)
            builtTrees.append(IsolationTree(root: root))
        }
        
        // Calculate scores for training set to determine score threshold for contamination
        let scores = calculateScores(trees: builtTrees, data: data, subsampleSize: subsampleSize)
        let sortedScores = scores.sorted(by: >)
        let cutoffIdx = Int(Double(numSamples) * contamination)
        let threshold = sortedScores[min(cutoffIdx, numSamples - 1)]
        
        return IsolationForest(trees: builtTrees, thresholdScore: threshold, nEstimators: nEstimators, maxSamples: maxSamples, contamination: contamination)
    }
    
    /// Predicts anomaly labels (1: inlier, -1: outlier) and anomaly scores for new features.
    public func predict(data: [[Double]]) throws -> AnomalyPrediction {
        guard !data.isEmpty else { throw PreprocessingError.emptyInput }
        let subsampleSize = min(maxSamples ?? min(256, data.count), data.count)
        let scores = Self.calculateScores(trees: trees, data: data, subsampleSize: subsampleSize)
        let labels = scores.map { $0 >= thresholdScore ? -1 : 1 }
        return AnomalyPrediction(labels: labels, scores: scores)
    }
    
    private static func buildTree(data: [[Double]], currentDepth: Int, maxDepth: Int, numFeatures: Int, rng: inout SeededRandom) -> Node {
        let node = Node()
        node.size = data.count
        if currentDepth >= maxDepth || data.count <= 1 {
            return node
        }
        
        let featIdx = rng.nextInt(upperBound: numFeatures)
        let values = data.map { $0[featIdx] }
        guard let minVal = values.min(), let maxVal = values.max(), minVal < maxVal else {
            return node
        }
        
        let ratio = Double(rng.nextInt(upperBound: 10000)) / 10000.0
        let splitVal = minVal + ratio * (maxVal - minVal)
        node.featureIndex = featIdx
        node.splitValue = splitVal
        
        let leftData = data.filter { $0[featIdx] < splitVal }
        let rightData = data.filter { $0[featIdx] >= splitVal }
        
        if leftData.isEmpty || rightData.isEmpty {
            return node
        }
        
        node.left = buildTree(data: leftData, currentDepth: currentDepth + 1, maxDepth: maxDepth, numFeatures: numFeatures, rng: &rng)
        node.right = buildTree(data: rightData, currentDepth: currentDepth + 1, maxDepth: maxDepth, numFeatures: numFeatures, rng: &rng)
        return node
    }
    
    private static func calculateScores(trees: [IsolationTree], data: [[Double]], subsampleSize: Int) -> [Double] {
        let cSubsample = averagePathLength(n: subsampleSize)
        guard cSubsample > 0 else { return [Double](repeating: 0.5, count: data.count) }
        
        return data.map { sample in
            let totalPath = trees.reduce(0.0) { sum, tree in
                sum + pathLength(sample: sample, node: tree.root, currentDepth: 0)
            }
            let avgPath = totalPath / Double(trees.count)
            return pow(2.0, -avgPath / cSubsample)
        }
    }
    
    private static func pathLength(sample: [Double], node: Node, currentDepth: Int) -> Double {
        if node.isLeaf {
            return Double(currentDepth) + averagePathLength(n: node.size)
        }
        if sample[node.featureIndex] < node.splitValue {
            return pathLength(sample: sample, node: node.left!, currentDepth: currentDepth + 1)
        } else {
            return pathLength(sample: sample, node: node.right!, currentDepth: currentDepth + 1)
        }
    }
    
    private static func averagePathLength(n: Int) -> Double {
        if n <= 1 { return 0.0 }
        if n == 2 { return 1.0 }
        let eulerConstant = 0.5772156649
        return 2.0 * (log(Double(n - 1)) + eulerConstant) - (2.0 * Double(n - 1) / Double(n))
    }
}

/// Local Outlier Factor (LOF) algorithm for local density-based anomaly detection.
public final class LocalOutlierFactor: Sendable {
    public let k: Int
    public let contamination: Double
    
    public init(k: Int = 20, contamination: Double = 0.1) {
        self.k = k
        self.contamination = max(0.0, min(0.5, contamination))
    }
    
    /// Computes Local Outlier Factors for dataset.
    public func fitPredict(data: [[Double]]) throws -> AnomalyPrediction {
        guard data.count > k else {
            throw PreprocessingError.emptyInput
        }
        let numSamples = data.count
        let numFeatures = data[0].count
        
        // 1. Distance matrix
        var distMatrix = [[Double]](repeating: [Double](repeating: 0.0, count: numSamples), count: numSamples)
        for i in 0..<numSamples {
            for j in (i+1)..<numSamples {
                var d = 0.0
                for f in 0..<numFeatures {
                    let diff = data[i][f] - data[j][f]
                    d += diff * diff
                }
                d = sqrt(d)
                distMatrix[i][j] = d
                distMatrix[j][i] = d
            }
        }
        
        // 2. k-distance and k-neighbors for each sample
        var kDistances = [Double](repeating: 0.0, count: numSamples)
        var kNeighbors = [[Int]](repeating: [], count: numSamples)
        
        for i in 0..<numSamples {
            let sortedIdx = (0..<numSamples).filter { $0 != i }.sorted { distMatrix[i][$0] < distMatrix[i][$1] }
            let neighbors = Array(sortedIdx.prefix(k))
            kNeighbors[i] = neighbors
            kDistances[i] = distMatrix[i][neighbors.last!]
        }
        
        // 3. Local Reachability Density (LRD)
        var lrd = [Double](repeating: 0.0, count: numSamples)
        for i in 0..<numSamples {
            var sumReachDist = 0.0
            for n in kNeighbors[i] {
                let reachDist = max(distMatrix[i][n], kDistances[n])
                sumReachDist += reachDist
            }
            lrd[i] = sumReachDist > 0 ? Double(kNeighbors[i].count) / sumReachDist : 0.0
        }
        
        // 4. Local Outlier Factor (LOF)
        var lofScores = [Double](repeating: 1.0, count: numSamples)
        for i in 0..<numSamples {
            guard lrd[i] > 0 else { continue }
            let sumLrdRatio = kNeighbors[i].reduce(0.0) { $0 + (lrd[$1] / lrd[i]) }
            lofScores[i] = sumLrdRatio / Double(kNeighbors[i].count)
        }
        
        let sortedLof = lofScores.sorted(by: >)
        let cutoffIdx = Int(Double(numSamples) * contamination)
        let threshold = sortedLof[min(cutoffIdx, numSamples - 1)]
        let labels = lofScores.map { $0 >= threshold ? -1 : 1 }
        
        return AnomalyPrediction(labels: labels, scores: lofScores)
    }
}
