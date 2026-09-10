import Foundation

// MARK: - Tree Node Representation for Histogram GBDT

/// A node in a histogram-binned decision tree.
public struct HistTreeNode: Sendable {
    /// Index of the feature split.
    public let featureIndex: Int
    /// Discrete integer bin threshold for the split (`<= binThreshold` goes to left child).
    public let binThreshold: UInt8
    /// Continuous threshold value corresponding to the bin boundary.
    public let continuousThreshold: Double
    /// Index of the left child in the flat tree array.
    public let leftChild: Int
    /// Index of the right child in the flat tree array.
    public let rightChild: Int
    /// Leaf update value applied during boosting.
    public let value: Double
    /// Indicates whether this node is a terminal leaf.
    public let isLeaf: Bool

    /// Creates a new histogram tree node.
    ///
    /// - Parameters:
    ///   - featureIndex: Feature column index.
    ///   - binThreshold: Discrete bin split threshold.
    ///   - continuousThreshold: Continuous value threshold.
    ///   - leftChild: Left child array index.
    ///   - rightChild: Right child array index.
    ///   - value: Prediction delta value for leaves.
    ///   - isLeaf: True if this is a terminal leaf.
    public init(
        featureIndex: Int,
        binThreshold: UInt8,
        continuousThreshold: Double,
        leftChild: Int,
        rightChild: Int,
        value: Double,
        isLeaf: Bool
    ) {
        self.featureIndex = featureIndex
        self.binThreshold = binThreshold
        self.continuousThreshold = continuousThreshold
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.value = value
        self.isLeaf = isLeaf
    }
}

// MARK: - HistGradientBoostingClassifier

/// Fast histogram-binned gradient boosted decision trees for large-scale tabular classification.
///
/// ## Optimization
/// Discretizes continuous features into 256 integer bins, accelerating split evaluations from \(O(N \log N)\) to \(O(K)\).
/// Instead of sorting continuous feature values at every node, feature distributions are mapped into discrete `UInt8`
/// bins. Gradients and Hessians are accumulated into compact 256-element histograms, enabling sub-millisecond split discovery.
///
/// ## Thread Safety
/// Implemented as an isolated Swift actor guaranteeing data-race freedom across concurrent tasks.
public actor HistGradientBoostingClassifier: ClassifierEstimator {

    /// Number of boosting iterations (trees).
    public let nEstimators: Int
    /// Shrinkage parameter scaling the contribution of each tree.
    public let learningRate: Double
    /// Maximum depth of individual trees.
    public let maxDepth: Int
    /// Maximum number of discrete bins for continuous features (at most 256).
    public let maxBins: Int
    /// Minimum number of samples required to form a leaf node.
    public let minSamplesLeaf: Int
    /// L2 regularization term applied to leaf weights (\(\lambda\)).
    public let l2Regularization: Double
    /// Optional random seed for reproducible training.
    public let randomState: Int?

    /// Bin upper thresholds for each feature column: `binEdges[featureIdx]` has up to `maxBins` values.
    private var binEdges: [[Double]] = []
    /// Trained ensemble of flat trees.
    private var trees: [[HistTreeNode]] = []
    /// Initial baseline log-odds prediction.
    private var initialLogOdds: Double = 0.0
    /// Unique class labels observed during fitting.
    private var classes: [Int] = []

    /// Creates a new histogram gradient boosted classifier.
    ///
    /// - Parameters:
    ///   - nEstimators: Number of boosting iterations. Default is `100`.
    ///   - learningRate: Shrinkage parameter. Default is `0.1`.
    ///   - maxDepth: Maximum depth of decision trees. Default is `5`.
    ///   - maxBins: Maximum number of discrete bins (between 2 and 256). Default is `256`.
    ///   - minSamplesLeaf: Minimum samples per leaf. Default is `20`.
    ///   - l2Regularization: L2 regularization parameter (\(\lambda\)). Default is `1.0`.
    ///   - randomState: Random state seed. Default is `nil`.
    /// - Throws: `SwiftMLError.invalidParameter` if any hyperparameter is non-positive or out of range.
    ///
    /// ## Thread Safety
    /// Thread-safe via actor isolation.
    ///
    /// ## Complexity
    /// \(O(1)\) initialization.
    public init(
        nEstimators: Int = 100,
        learningRate: Double = 0.1,
        maxDepth: Int = 5,
        maxBins: Int = 256,
        minSamplesLeaf: Int = 20,
        l2Regularization: Double = 1.0,
        randomState: Int? = nil
    ) throws {
        guard nEstimators > 0 else { throw SwiftMLError.invalidParameter("nEstimators must be > 0") }
        guard learningRate > 0.0 else { throw SwiftMLError.invalidParameter("learningRate must be > 0") }
        guard maxDepth > 0 else { throw SwiftMLError.invalidParameter("maxDepth must be > 0") }
        guard maxBins >= 2 && maxBins <= 256 else { throw SwiftMLError.invalidParameter("maxBins must be in 2...256") }
        guard minSamplesLeaf > 0 else { throw SwiftMLError.invalidParameter("minSamplesLeaf must be > 0") }
        guard l2Regularization >= 0.0 else { throw SwiftMLError.invalidParameter("l2Regularization must be >= 0") }

        self.nEstimators = nEstimators
        self.learningRate = learningRate
        self.maxDepth = maxDepth
        self.maxBins = maxBins
        self.minSamplesLeaf = minSamplesLeaf
        self.l2Regularization = l2Regularization
        self.randomState = randomState
    }

    /// Fits the histogram GBDT classifier on training features and labels.
    ///
    /// - Parameters:
    ///   - features: 2D feature matrix of shape `[N, P]`.
    ///   - targets: Array of integer or floating-point class labels.
    /// - Throws: `SwiftMLError` if inputs are empty or dimensions mismatch.
    ///
    /// ## Thread Safety
    /// Thread-safe via actor isolation.
    ///
    /// ## Complexity
    /// \(O(P \cdot N \log N + M \cdot (N + P \cdot K \cdot 2^{\text{maxDepth}}))\) where \(K \le 256\).
    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty else { throw SwiftMLError.emptyInput }
        guard features.count == targets.count else {
            throw SwiftMLError.dimensionMismatch(expected: features.count, got: targets.count)
        }

        let n = features.count
        let numFeatures = features[0].count
        guard numFeatures > 0 else { throw SwiftMLError.invalidParameter("Features must have at least 1 column") }

        let uniqueClasses = Array(Set(targets.map { Int($0) })).sorted()
        guard uniqueClasses.count >= 2 else {
            throw SwiftMLError.invalidParameter("Classifier requires at least 2 distinct classes")
        }
        self.classes = uniqueClasses

        // 1. Build bin edges for each feature column
        var computedEdges: [[Double]] = []
        computedEdges.reserveCapacity(numFeatures)

        for colIdx in 0..<numFeatures {
            var colVals = [Double](repeating: 0.0, count: n)
            for rowIdx in 0..<n {
                colVals[rowIdx] = features[rowIdx][colIdx]
            }
            colVals.sort()

            // Find unique split quantiles
            var edges: [Double] = []
            let numUnique = Set(colVals).count
            let k = min(maxBins, numUnique)
            if k <= 1 {
                edges = [colVals[0]]
            } else {
                for b in 1..<k {
                    let quantileIdx = Int((Double(b) / Double(k)) * Double(n - 1))
                    let val = colVals[quantileIdx]
                    if edges.isEmpty || val > edges.last! {
                        edges.append(val)
                    }
                }
            }
            computedEdges.append(edges)
        }
        self.binEdges = computedEdges

        // 2. Discretize features into UInt8 binned matrix [N, P]
        var binnedMatrix: [[UInt8]] = Array(repeating: [UInt8](repeating: 0, count: numFeatures), count: n)
        for rowIdx in 0..<n {
            let row = features[rowIdx]
            for colIdx in 0..<numFeatures {
                binnedMatrix[rowIdx][colIdx] = Self.binValue(row[colIdx], edges: computedEdges[colIdx])
            }
        }

        // 3. Binary classification setup (class 1 vs class 0)
        let positiveClass = uniqueClasses[1]
        let yBinary: [Double] = targets.map { Int($0) == positiveClass ? 1.0 : 0.0 }
        let posCount = yBinary.reduce(0.0, +)
        let priorProb = max(1e-5, min(1.0 - 1e-5, posCount / Double(n)))
        self.initialLogOdds = log(priorProb / (1.0 - priorProb))

        var currentLogOdds = [Double](repeating: initialLogOdds, count: n)
        var ensemble: [[HistTreeNode]] = []
        ensemble.reserveCapacity(nEstimators)

        for _ in 0..<nEstimators {
            // Compute negative gradients and hessians for logistic loss
            var gradients = [Double](repeating: 0.0, count: n)
            var hessians = [Double](repeating: 0.0, count: n)

            for i in 0..<n {
                let p = 1.0 / (1.0 + exp(-currentLogOdds[i]))
                gradients[i] = yBinary[i] - p
                hessians[i] = max(1e-4, p * (1.0 - p))
            }

            // Build histogram tree
            var treeNodes: [HistTreeNode] = []
            let allIndices = Array(0..<n)
            _ = buildHistTree(
                binnedMatrix: binnedMatrix,
                binEdges: computedEdges,
                gradients: gradients,
                hessians: hessians,
                sampleIndices: allIndices,
                depth: 0,
                treeNodes: &treeNodes
            )

            // Update current predictions
            for i in 0..<n {
                let delta = Self.predictBinnedSample(binnedMatrix[i], nodes: treeNodes)
                currentLogOdds[i] += learningRate * delta
            }

            ensemble.append(treeNodes)
        }

        self.trees = ensemble
    }

    /// Predicts discrete class labels for the input feature matrix.
    ///
    /// - Parameter features: 2D array of samples to evaluate.
    /// - Returns: Predicted integer class label for each sample.
    /// - Throws: `SwiftMLError.notFitted` if `fit()` has not been invoked.
    ///
    /// ## Thread Safety
    /// Thread-safe via actor isolation.
    public func predict(features: [[Double]]) async throws -> [Int] {
        guard !trees.isEmpty, !classes.isEmpty else { throw SwiftMLError.notFitted }
        guard !features.isEmpty else { return [] }

        let probabilities = try await predictProbability(features: features)
        let positiveClass = classes.count > 1 ? classes[1] : classes[0]
        let negativeClass = classes[0]

        return probabilities.map { probRow in
            let posProb = probRow[1]
            return posProb >= 0.5 ? positiveClass : negativeClass
        }
    }

    /// Predicts class probabilities for the input feature matrix.
    ///
    /// - Parameter features: 2D array of samples to evaluate.
    /// - Returns: 2D array where each row contains `[P(class 0), P(class 1)]`.
    /// - Throws: `SwiftMLError.notFitted` if the model has not been trained.
    ///
    /// ## Thread Safety
    /// Thread-safe via actor isolation.
    public func predictProbability(features: [[Double]]) async throws -> [[Double]] {
        guard !trees.isEmpty, !classes.isEmpty else { throw SwiftMLError.notFitted }
        guard !features.isEmpty else { return [] }

        var results: [[Double]] = []
        results.reserveCapacity(features.count)

        for row in features {
            var logOdds = initialLogOdds
            // Discretize row
            let binnedRow = (0..<binEdges.count).map { col in
                Self.binValue(row[col], edges: binEdges[col])
            }
            for tree in trees {
                let delta = Self.predictBinnedSample(binnedRow, nodes: tree)
                logOdds += learningRate * delta
            }
            let p1 = 1.0 / (1.0 + exp(-logOdds))
            let p0 = 1.0 - p1
            results.append([p0, p1])
        }

        return results
    }

    // MARK: - Tree Construction via 256-Bin Histograms

    private func buildHistTree(
        binnedMatrix: [[UInt8]],
        binEdges: [[Double]],
        gradients: [Double],
        hessians: [Double],
        sampleIndices: [Int],
        depth: Int,
        treeNodes: inout [HistTreeNode]
    ) -> Int {
        var sumG = 0.0
        var sumH = 0.0
        for idx in sampleIndices {
            sumG += gradients[idx]
            sumH += hessians[idx]
        }

        let leafValue = sumG / (sumH + l2Regularization)

        // Stopping conditions: maxDepth reached or insufficient samples
        if depth >= maxDepth || sampleIndices.count < 2 * minSamplesLeaf {
            let leaf = HistTreeNode(
                featureIndex: -1,
                binThreshold: 0,
                continuousThreshold: 0.0,
                leftChild: -1,
                rightChild: -1,
                value: leafValue,
                isLeaf: true
            )
            treeNodes.append(leaf)
            return treeNodes.count - 1
        }

        // Find best split using O(K) histogram evaluation
        var bestGain = 0.0
        var bestFeature = -1
        var bestBin: UInt8 = 0
        var bestContinuousThreshold = 0.0

        let baseScore = (sumG * sumG) / (sumH + l2Regularization)
        let numFeatures = binEdges.count

        for colIdx in 0..<numFeatures {
            let edges = binEdges[colIdx]
            let numBins = edges.count + 1
            guard numBins >= 2 else { continue }

            var histG = [Double](repeating: 0.0, count: numBins)
            var histH = [Double](repeating: 0.0, count: numBins)
            var histCount = [Int](repeating: 0, count: numBins)

            // Accumulate histograms in one linear pass over node samples
            for idx in sampleIndices {
                let bin = Int(binnedMatrix[idx][colIdx])
                histG[bin] += gradients[idx]
                histH[bin] += hessians[idx]
                histCount[bin] += 1
            }

            // Linear scan over bins
            var leftG = 0.0
            var leftH = 0.0
            var leftCount = 0

            for b in 0..<(numBins - 1) {
                leftG += histG[b]
                leftH += histH[b]
                leftCount += histCount[b]

                let rightG = sumG - leftG
                let rightH = sumH - leftH
                let rightCount = sampleIndices.count - leftCount

                if leftCount < minSamplesLeaf || rightCount < minSamplesLeaf {
                    continue
                }

                let leftScore = (leftG * leftG) / (leftH + l2Regularization)
                let rightScore = (rightG * rightG) / (rightH + l2Regularization)
                let gain = 0.5 * (leftScore + rightScore - baseScore)

                if gain > bestGain {
                    bestGain = gain
                    bestFeature = colIdx
                    bestBin = UInt8(b)
                    bestContinuousThreshold = edges[b]
                }
            }
        }

        guard bestFeature != -1 else {
            let leaf = HistTreeNode(
                featureIndex: -1,
                binThreshold: 0,
                continuousThreshold: 0.0,
                leftChild: -1,
                rightChild: -1,
                value: leafValue,
                isLeaf: true
            )
            treeNodes.append(leaf)
            return treeNodes.count - 1
        }

        // Partition samples
        var leftIndices: [Int] = []
        var rightIndices: [Int] = []
        leftIndices.reserveCapacity(sampleIndices.count / 2)
        rightIndices.reserveCapacity(sampleIndices.count / 2)

        for idx in sampleIndices {
            if binnedMatrix[idx][bestFeature] <= bestBin {
                leftIndices.append(idx)
            } else {
                rightIndices.append(idx)
            }
        }

        let currentIndex = treeNodes.count
        // Reserve slot
        treeNodes.append(HistTreeNode(
            featureIndex: -1,
            binThreshold: 0,
            continuousThreshold: 0.0,
            leftChild: -1,
            rightChild: -1,
            value: 0.0,
            isLeaf: false
        ))

        let leftChildIdx = buildHistTree(
            binnedMatrix: binnedMatrix,
            binEdges: binEdges,
            gradients: gradients,
            hessians: hessians,
            sampleIndices: leftIndices,
            depth: depth + 1,
            treeNodes: &treeNodes
        )
        let rightChildIdx = buildHistTree(
            binnedMatrix: binnedMatrix,
            binEdges: binEdges,
            gradients: gradients,
            hessians: hessians,
            sampleIndices: rightIndices,
            depth: depth + 1,
            treeNodes: &treeNodes
        )

        treeNodes[currentIndex] = HistTreeNode(
            featureIndex: bestFeature,
            binThreshold: bestBin,
            continuousThreshold: bestContinuousThreshold,
            leftChild: leftChildIdx,
            rightChild: rightChildIdx,
            value: leafValue,
            isLeaf: false
        )

        return currentIndex
    }

    // MARK: - Helpers

    private static func binValue(_ value: Double, edges: [Double]) -> UInt8 {
        var low = 0
        var high = edges.count
        while low < high {
            let mid = (low + high) / 2
            if edges[mid] < value {
                low = mid + 1
            } else {
                high = mid
            }
        }
        return UInt8(min(low, 255))
    }

    private static func predictBinnedSample(_ binnedSample: [UInt8], nodes: [HistTreeNode]) -> Double {
        guard !nodes.isEmpty else { return 0.0 }
        var curr = 0
        while !nodes[curr].isLeaf {
            let node = nodes[curr]
            if binnedSample[node.featureIndex] <= node.binThreshold {
                curr = node.leftChild
            } else {
                curr = node.rightChild
            }
        }
        return nodes[curr].value
    }
}

// MARK: - HistGradientBoostingRegressor

/// Fast histogram-binned gradient boosted decision trees for large-scale tabular regression.
///
/// ## Optimization
/// Evaluates split candidates in \(O(K)\) time across 256 feature bins, minimizing squared error loss.
///
/// ## Thread Safety
/// Implemented as an isolated Swift actor guaranteeing data-race freedom across concurrent tasks.
public actor HistGradientBoostingRegressor: RegressorEstimator {

    /// Number of boosting iterations.
    public let nEstimators: Int
    /// Shrinkage parameter.
    public let learningRate: Double
    /// Maximum depth of regression trees.
    public let maxDepth: Int
    /// Maximum discrete bins for continuous features.
    public let maxBins: Int
    /// Minimum samples required per leaf.
    public let minSamplesLeaf: Int
    /// L2 regularization parameter (\(\lambda\)).
    public let l2Regularization: Double

    private var binEdges: [[Double]] = []
    private var trees: [[HistTreeNode]] = []
    private var initialMean: Double = 0.0

    /// Creates a new histogram gradient boosted regressor.
    ///
    /// - Parameters:
    ///   - nEstimators: Boosting iterations. Default is `100`.
    ///   - learningRate: Shrinkage parameter. Default is `0.1`.
    ///   - maxDepth: Maximum tree depth. Default is `5`.
    ///   - maxBins: Discrete bins (between 2 and 256). Default is `256`.
    ///   - minSamplesLeaf: Minimum samples per leaf. Default is `20`.
    ///   - l2Regularization: L2 regularization term. Default is `1.0`.
    /// - Throws: `SwiftMLError.invalidParameter` if parameters are out of range.
    public init(
        nEstimators: Int = 100,
        learningRate: Double = 0.1,
        maxDepth: Int = 5,
        maxBins: Int = 256,
        minSamplesLeaf: Int = 20,
        l2Regularization: Double = 1.0
    ) throws {
        guard nEstimators > 0 else { throw SwiftMLError.invalidParameter("nEstimators must be > 0") }
        guard learningRate > 0.0 else { throw SwiftMLError.invalidParameter("learningRate must be > 0") }
        guard maxDepth > 0 else { throw SwiftMLError.invalidParameter("maxDepth must be > 0") }
        guard maxBins >= 2 && maxBins <= 256 else { throw SwiftMLError.invalidParameter("maxBins must be in 2...256") }
        guard minSamplesLeaf > 0 else { throw SwiftMLError.invalidParameter("minSamplesLeaf must be > 0") }
        guard l2Regularization >= 0.0 else { throw SwiftMLError.invalidParameter("l2Regularization must be >= 0") }

        self.nEstimators = nEstimators
        self.learningRate = learningRate
        self.maxDepth = maxDepth
        self.maxBins = maxBins
        self.minSamplesLeaf = minSamplesLeaf
        self.l2Regularization = l2Regularization
    }

    /// Fits the histogram GBDT regressor on features and continuous targets.
    ///
    /// - Parameters:
    ///   - features: 2D feature array.
    ///   - targets: 1D array of target continuous values.
    /// - Throws: `SwiftMLError` if inputs are invalid.
    public func fit(features: [[Double]], targets: [Double]) async throws {
        guard !features.isEmpty else { throw SwiftMLError.emptyInput }
        guard features.count == targets.count else {
            throw SwiftMLError.dimensionMismatch(expected: features.count, got: targets.count)
        }

        let n = features.count
        let numFeatures = features[0].count

        // 1. Build bin edges
        var computedEdges: [[Double]] = []
        computedEdges.reserveCapacity(numFeatures)

        for colIdx in 0..<numFeatures {
            var colVals = [Double](repeating: 0.0, count: n)
            for rowIdx in 0..<n {
                colVals[rowIdx] = features[rowIdx][colIdx]
            }
            colVals.sort()

            var edges: [Double] = []
            let numUnique = Set(colVals).count
            let k = min(maxBins, numUnique)
            if k <= 1 {
                edges = [colVals[0]]
            } else {
                for b in 1..<k {
                    let quantileIdx = Int((Double(b) / Double(k)) * Double(n - 1))
                    let val = colVals[quantileIdx]
                    if edges.isEmpty || val > edges.last! {
                        edges.append(val)
                    }
                }
            }
            computedEdges.append(edges)
        }
        self.binEdges = computedEdges

        // 2. Discretize
        var binnedMatrix: [[UInt8]] = Array(repeating: [UInt8](repeating: 0, count: numFeatures), count: n)
        for rowIdx in 0..<n {
            let row = features[rowIdx]
            for colIdx in 0..<numFeatures {
                binnedMatrix[rowIdx][colIdx] = Self.binValue(row[colIdx], edges: computedEdges[colIdx])
            }
        }

        self.initialMean = targets.reduce(0.0, +) / Double(n)
        var currentPredictions = [Double](repeating: initialMean, count: n)
        var ensemble: [[HistTreeNode]] = []
        ensemble.reserveCapacity(nEstimators)

        for _ in 0..<nEstimators {
            // Negative gradient of MSE: targets - currentPredictions, hessians = 1.0
            var gradients = [Double](repeating: 0.0, count: n)
            let hessians = [Double](repeating: 1.0, count: n)

            for i in 0..<n {
                gradients[i] = targets[i] - currentPredictions[i]
            }

            var treeNodes: [HistTreeNode] = []
            let allIndices = Array(0..<n)
            _ = buildHistTree(
                binnedMatrix: binnedMatrix,
                binEdges: computedEdges,
                gradients: gradients,
                hessians: hessians,
                sampleIndices: allIndices,
                depth: 0,
                treeNodes: &treeNodes
            )

            for i in 0..<n {
                let delta = Self.predictBinnedSample(binnedMatrix[i], nodes: treeNodes)
                currentPredictions[i] += learningRate * delta
            }

            ensemble.append(treeNodes)
        }

        self.trees = ensemble
    }

    /// Predicts continuous target values for the given features.
    ///
    /// - Parameter features: 2D array of samples.
    /// - Returns: Array of predicted values.
    /// - Throws: `SwiftMLError.notFitted` if the model has not been trained.
    public func predict(features: [[Double]]) async throws -> [Double] {
        guard !trees.isEmpty else { throw SwiftMLError.notFitted }
        guard !features.isEmpty else { return [] }

        var results = [Double](repeating: initialMean, count: features.count)
        for i in 0..<features.count {
            let row = features[i]
            let binnedRow = (0..<binEdges.count).map { col in
                Self.binValue(row[col], edges: binEdges[col])
            }
            for tree in trees {
                let delta = Self.predictBinnedSample(binnedRow, nodes: tree)
                results[i] += learningRate * delta
            }
        }
        return results
    }

    // MARK: - Tree Construction Details

    private func buildHistTree(
        binnedMatrix: [[UInt8]],
        binEdges: [[Double]],
        gradients: [Double],
        hessians: [Double],
        sampleIndices: [Int],
        depth: Int,
        treeNodes: inout [HistTreeNode]
    ) -> Int {
        var sumG = 0.0
        var sumH = 0.0
        for idx in sampleIndices {
            sumG += gradients[idx]
            sumH += hessians[idx]
        }

        let leafValue = sumG / (sumH + l2Regularization)

        if depth >= maxDepth || sampleIndices.count < 2 * minSamplesLeaf {
            let leaf = HistTreeNode(
                featureIndex: -1,
                binThreshold: 0,
                continuousThreshold: 0.0,
                leftChild: -1,
                rightChild: -1,
                value: leafValue,
                isLeaf: true
            )
            treeNodes.append(leaf)
            return treeNodes.count - 1
        }

        var bestGain = 0.0
        var bestFeature = -1
        var bestBin: UInt8 = 0
        var bestContinuousThreshold = 0.0

        let baseScore = (sumG * sumG) / (sumH + l2Regularization)
        let numFeatures = binEdges.count

        for colIdx in 0..<numFeatures {
            let edges = binEdges[colIdx]
            let numBins = edges.count + 1
            guard numBins >= 2 else { continue }

            var histG = [Double](repeating: 0.0, count: numBins)
            var histH = [Double](repeating: 0.0, count: numBins)
            var histCount = [Int](repeating: 0, count: numBins)

            for idx in sampleIndices {
                let bin = Int(binnedMatrix[idx][colIdx])
                histG[bin] += gradients[idx]
                histH[bin] += hessians[idx]
                histCount[bin] += 1
            }

            var leftG = 0.0
            var leftH = 0.0
            var leftCount = 0

            for b in 0..<(numBins - 1) {
                leftG += histG[b]
                leftH += histH[b]
                leftCount += histCount[b]

                let rightG = sumG - leftG
                let rightH = sumH - leftH
                let rightCount = sampleIndices.count - leftCount

                if leftCount < minSamplesLeaf || rightCount < minSamplesLeaf {
                    continue
                }

                let leftScore = (leftG * leftG) / (leftH + l2Regularization)
                let rightScore = (rightG * rightG) / (rightH + l2Regularization)
                let gain = 0.5 * (leftScore + rightScore - baseScore)

                if gain > bestGain {
                    bestGain = gain
                    bestFeature = colIdx
                    bestBin = UInt8(b)
                    bestContinuousThreshold = edges[b]
                }
            }
        }

        guard bestFeature != -1 else {
            let leaf = HistTreeNode(
                featureIndex: -1,
                binThreshold: 0,
                continuousThreshold: 0.0,
                leftChild: -1,
                rightChild: -1,
                value: leafValue,
                isLeaf: true
            )
            treeNodes.append(leaf)
            return treeNodes.count - 1
        }

        var leftIndices: [Int] = []
        var rightIndices: [Int] = []
        leftIndices.reserveCapacity(sampleIndices.count / 2)
        rightIndices.reserveCapacity(sampleIndices.count / 2)

        for idx in sampleIndices {
            if binnedMatrix[idx][bestFeature] <= bestBin {
                leftIndices.append(idx)
            } else {
                rightIndices.append(idx)
            }
        }

        let currentIndex = treeNodes.count
        treeNodes.append(HistTreeNode(
            featureIndex: -1,
            binThreshold: 0,
            continuousThreshold: 0.0,
            leftChild: -1,
            rightChild: -1,
            value: 0.0,
            isLeaf: false
        ))

        let leftChildIdx = buildHistTree(
            binnedMatrix: binnedMatrix,
            binEdges: binEdges,
            gradients: gradients,
            hessians: hessians,
            sampleIndices: leftIndices,
            depth: depth + 1,
            treeNodes: &treeNodes
        )
        let rightChildIdx = buildHistTree(
            binnedMatrix: binnedMatrix,
            binEdges: binEdges,
            gradients: gradients,
            hessians: hessians,
            sampleIndices: rightIndices,
            depth: depth + 1,
            treeNodes: &treeNodes
        )

        treeNodes[currentIndex] = HistTreeNode(
            featureIndex: bestFeature,
            binThreshold: bestBin,
            continuousThreshold: bestContinuousThreshold,
            leftChild: leftChildIdx,
            rightChild: rightChildIdx,
            value: leafValue,
            isLeaf: false
        )

        return currentIndex
    }

    private static func binValue(_ value: Double, edges: [Double]) -> UInt8 {
        var low = 0
        var high = edges.count
        while low < high {
            let mid = (low + high) / 2
            if edges[mid] < value {
                low = mid + 1
            } else {
                high = mid
            }
        }
        return UInt8(min(low, 255))
    }

    private static func predictBinnedSample(_ binnedSample: [UInt8], nodes: [HistTreeNode]) -> Double {
        guard !nodes.isEmpty else { return 0.0 }
        var curr = 0
        while !nodes[curr].isLeaf {
            let node = nodes[curr]
            if binnedSample[node.featureIndex] <= node.binThreshold {
                curr = node.leftChild
            } else {
                curr = node.rightChild
            }
        }
        return nodes[curr].value
    }
}
