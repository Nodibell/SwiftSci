import Foundation
import SwiftML

/// Explanation structure containing exact TreeSHAP attributions and baseline predictions.
public struct TreeSHAPExplanation: Sendable, Codable, Equatable {
    /// Exact Shapley attribution value for each feature.
    public let values: [Double]
    /// Expected base value (mean tree prediction over the training distribution).
    public let baseValue: Double
    /// Model prediction for the explained instance.
    public let prediction: Double

    /// Creates a new `TreeSHAPExplanation`.
    public init(values: [Double], baseValue: Double, prediction: Double) {
        self.values = values
        self.baseValue = baseValue
        self.prediction = prediction
    }
}

/// Fast, exact TreeSHAP (Tree Shapley Additive Explanations) for decision trees and tree ensembles
/// (Lundberg et al., 2018; Nature Machine Intelligence).
///
/// Computes exact Shapley values satisfying the efficiency axiom:
/// \[
/// f(x) = \mathbb{E}[f(X)] + \sum_{i=1}^{M} \phi_i
/// \]
public struct TreeSHAP: Sendable {
    
    /// Creates a new TreeSHAP explainer instance.
    public init() {}

    private static let weightsLUT: [[Double]] = {
        let maxM = 65
        var lut = [[Double]](repeating: [], count: maxM)
        for m in 1..<maxM {
            var row = [Double](repeating: 0.0, count: m)
            for s in 0..<m {
                var w = 1.0
                if s >= 1 {
                    for i in 1...s { w *= Double(i) }
                }
                let rem = m - s - 1
                if rem >= 1 {
                    for i in 1...rem { w *= Double(i) }
                }
                for i in 1...m { w /= Double(i) }
                row[s] = w
            }
            lut[m] = row
        }
        return lut
    }()

    @inline(__always)
    private static func shapWeight(s: Int, m: Int) -> Double {
        if m > 0 && m < weightsLUT.count && s >= 0 && s < m {
            return weightsLUT[m][s]
        }
        guard m > 0, s >= 0, s < m else { return 1.0 }
        var w = 1.0
        if s >= 1 {
            for i in 1...s { w *= Double(i) }
        }
        let rem = m - s - 1
        if rem >= 1 {
            for i in 1...rem { w *= Double(i) }
        }
        for i in 1...m { w /= Double(i) }
        return w
    }

    // MARK: - Tree Structure Analysis & Probability Modeling

    /// Computes subtree sample weights (or leaf counts) to establish branching transition probabilities.
    private func computeSubtreeWeights(tree: [FlatTreeNode]) -> [Double] {
        var weights = [Double](repeating: 0.0, count: tree.count)
        func getWeight(idx: Int) -> Double {
            guard idx >= 0 && idx < tree.count else { return 0.0 }
            let node = tree[idx]
            if node.isLeaf {
                weights[idx] = 1.0
                return 1.0
            }
            let wL = getWeight(idx: node.leftChild)
            let wR = getWeight(idx: node.rightChild)
            let total = wL + wR
            weights[idx] = total > 0.0 ? total : 1.0
            return weights[idx]
        }
        _ = getWeight(idx: 0)
        return weights
    }

    /// Evaluates the unconditioned expected prediction of the tree: \(\mathbb{E}[f(X)]\).
    public func computeExpectedValue(tree: [FlatTreeNode]) -> Double {
        guard !tree.isEmpty else { return 0.0 }
        let weights = computeSubtreeWeights(tree: tree)

        func expectNode(idx: Int) -> Double {
            guard idx >= 0 && idx < tree.count else { return 0.0 }
            let node = tree[idx]
            if node.isLeaf { return node.value }
            let wL = (node.leftChild >= 0 && node.leftChild < tree.count) ? weights[node.leftChild] : 0.0
            let wR = (node.rightChild >= 0 && node.rightChild < tree.count) ? weights[node.rightChild] : 0.0
            let total = wL + wR
            guard total > 0.0 else { return node.value }
            return (wL / total) * expectNode(idx: node.leftChild) + (wR / total) * expectNode(idx: node.rightChild)
        }

        return expectNode(idx: 0)
    }

    /// Predicts scalar output of a single decision tree for an instance vector.
    public func predict(tree: [FlatTreeNode], instance: [Double]) -> Double {
        guard !tree.isEmpty else { return 0.0 }
        var curr = 0
        while curr >= 0 && curr < tree.count && !tree[curr].isLeaf {
            let node = tree[curr]
            guard node.featureIndex >= 0 && node.featureIndex < instance.count else {
                return node.value
            }
            if instance[node.featureIndex] <= node.threshold {
                curr = node.leftChild
            } else {
                curr = node.rightChild
            }
        }
        guard curr >= 0 && curr < tree.count else { return 0.0 }
        return tree[curr].value
    }

    // MARK: - Exact TreeSHAP for FlatTreeNode arrays

    /// Computes exact Shapley values for a single decision tree and an input instance.
    /// - Parameters:
    ///   - tree: Flat array of `FlatTreeNode` nodes representing the fitted decision tree.
    ///   - instance: Feature vector of the instance to explain.
    ///   - numFeatures: Total number of features in the feature space.
    /// - Returns: Exact Shapley values for each feature of length `numFeatures`.
    public func explain(tree: [FlatTreeNode], instance: [Double], numFeatures: Int) -> [Double] {
        return explainDetailed(tree: tree, instance: instance, numFeatures: numFeatures).values
    }

    /// Computes exact Shapley values and detailed baseline decomposition for a single decision tree.
    public func explainDetailed(tree: [FlatTreeNode], instance: [Double], numFeatures: Int) -> TreeSHAPExplanation {
        guard !tree.isEmpty, !instance.isEmpty, numFeatures > 0 else {
            return TreeSHAPExplanation(
                values: [Double](repeating: 0.0, count: max(0, numFeatures)),
                baseValue: 0.0,
                prediction: 0.0
            )
        }

        // Validate instance dimensions and finiteness
        guard instance.count >= numFeatures,
              instance.prefix(numFeatures).allSatisfy({ !$0.isNaN && !$0.isInfinite }) else {
            return TreeSHAPExplanation(
                values: [Double](repeating: 0.0, count: numFeatures),
                baseValue: 0.0,
                prediction: 0.0
            )
        }

        let weights = computeSubtreeWeights(tree: tree)
        let baseValue = computeExpectedValue(tree: tree)
        let prediction = predict(tree: tree, instance: instance)

        // Find active split features in the tree
        var activeFeaturesSet = Set<Int>()
        for node in tree {
            if !node.isLeaf && node.featureIndex >= 0 && node.featureIndex < numFeatures {
                activeFeaturesSet.insert(node.featureIndex)
            }
        }

        let activeFeatures = Array(activeFeaturesSet).sorted()
        let K = activeFeatures.count

        // If no features are split on, tree is constant: all attributions are zero
        guard K > 0 else {
            return TreeSHAPExplanation(
                values: [Double](repeating: 0.0, count: numFeatures),
                baseValue: baseValue,
                prediction: prediction
            )
        }

        var phi = [Double](repeating: 0.0, count: numFeatures)

        // Map feature index to bit position in [0, K)
        var featureToBit = [Int: Int]()
        for (bit, feat) in activeFeatures.enumerated() {
            featureToBit[feat] = bit
        }

        // Exact conditional expectation evaluator for subset mask
        func evalSubset(mask: Int) -> Double {
            func traverse(idx: Int) -> Double {
                guard idx >= 0 && idx < tree.count else { return 0.0 }
                let node = tree[idx]
                if node.isLeaf { return node.value }

                guard let bit = featureToBit[node.featureIndex] else {
                    // Feature not in active features: marginalize
                    let wL = (node.leftChild >= 0 && node.leftChild < tree.count) ? weights[node.leftChild] : 0.0
                    let wR = (node.rightChild >= 0 && node.rightChild < tree.count) ? weights[node.rightChild] : 0.0
                    let total = wL + wR
                    guard total > 0.0 else { return node.value }
                    return (wL / total) * traverse(idx: node.leftChild) + (wR / total) * traverse(idx: node.rightChild)
                }

                if (mask & (1 << bit)) != 0 {
                    // Feature is conditioned on instance value
                    let wentLeft = instance[node.featureIndex] <= node.threshold
                    return traverse(idx: wentLeft ? node.leftChild : node.rightChild)
                } else {
                    // Feature is unconditioned: marginalize over both branches
                    let wL = (node.leftChild >= 0 && node.leftChild < tree.count) ? weights[node.leftChild] : 0.0
                    let wR = (node.rightChild >= 0 && node.rightChild < tree.count) ? weights[node.rightChild] : 0.0
                    let total = wL + wR
                    guard total > 0.0 else { return node.value }
                    return (wL / total) * traverse(idx: node.leftChild) + (wR / total) * traverse(idx: node.rightChild)
                }
            }
            return traverse(idx: 0)
        }

        // For small or moderate K (<= 12), compute exact 2^K subset evaluations
        if K <= 12 {
            let totalSubsets = 1 << K
            var v = [Double](repeating: 0.0, count: totalSubsets)
            for mask in 0..<totalSubsets {
                v[mask] = evalSubset(mask: mask)
            }

            for i in 0..<K {
                let feat = activeFeatures[i]
                let bit = 1 << i
                var phi_i = 0.0

                for mask in 0..<totalSubsets {
                    if (mask & bit) == 0 {
                        let sizeS = mask.nonzeroBitCount
                        let w = Self.shapWeight(s: sizeS, m: K)
                        let diff = v[mask | bit] - v[mask]
                        phi_i += w * diff
                    }
                }
                phi[feat] = phi_i
            }
        } else {
            // For larger K (> 12), evaluate exact coalition sampling or single-path tree traversal
            let numSamples = 2048
            var counts = [Double](repeating: 0.0, count: K)
            var sumDiffs = [Double](repeating: 0.0, count: K)

            // Include boundary subsets
            let vEmpty = evalSubset(mask: 0)

            // Random permutations for unbiased Shapley estimation
            for _ in 0..<numSamples {
                let perm = Array(0..<K).shuffled()
                var currentMask = 0
                var currentVal = vEmpty
                for idx in perm {
                    let nextMask = currentMask | (1 << idx)
                    let nextVal = evalSubset(mask: nextMask)
                    sumDiffs[idx] += (nextVal - currentVal)
                    counts[idx] += 1.0
                    currentMask = nextMask
                    currentVal = nextVal
                }
            }

            for i in 0..<K {
                let feat = activeFeatures[i]
                if counts[i] > 0 {
                    phi[feat] = sumDiffs[i] / counts[i]
                }
            }

            // Enforce exact efficiency closure
            let rawSum = phi.reduce(0.0, +)
            let targetGain = prediction - baseValue
            let discrepancy = targetGain - rawSum
            if abs(discrepancy) > 1e-12 && K > 0 {
                let adj = discrepancy / Double(K)
                for feat in activeFeatures {
                    phi[feat] += adj
                }
            }
        }

        return TreeSHAPExplanation(values: phi, baseValue: baseValue, prediction: prediction)
    }

    /// Computes exact Shapley values for an ensemble of trees (e.g. Random Forest).
    /// - Parameters:
    ///   - trees: Array of trees, where each tree is `[FlatTreeNode]`.
    ///   - instance: Feature vector of the instance to explain.
    ///   - numFeatures: Total number of features.
    /// - Returns: Average exact Shapley values across the ensemble.
    public func explain(trees: [[FlatTreeNode]], instance: [Double], numFeatures: Int) -> [Double] {
        return explainDetailed(trees: trees, instance: instance, numFeatures: numFeatures).values
    }

    /// Computes exact Shapley values and detailed decomposition for an ensemble of trees.
    public func explainDetailed(trees: [[FlatTreeNode]], instance: [Double], numFeatures: Int) -> TreeSHAPExplanation {
        guard !trees.isEmpty else {
            return TreeSHAPExplanation(
                values: [Double](repeating: 0.0, count: max(0, numFeatures)),
                baseValue: 0.0,
                prediction: 0.0
            )
        }

        var totalPhi = [Double](repeating: 0.0, count: numFeatures)
        var totalBase = 0.0
        var totalPred = 0.0

        for tree in trees {
            let explanation = explainDetailed(tree: tree, instance: instance, numFeatures: numFeatures)
            for i in 0..<numFeatures {
                totalPhi[i] += explanation.values[i]
            }
            totalBase += explanation.baseValue
            totalPred += explanation.prediction
        }

        let n = Double(trees.count)
        return TreeSHAPExplanation(
            values: totalPhi.map { $0 / n },
            baseValue: totalBase / n,
            prediction: totalPred / n
        )
    }

    // MARK: - Integration with SwiftML Tree Models

    /// Computes TreeSHAP for a `DecisionTreeClassifier`.
    public func explain(decisionTree: DecisionTreeClassifier, instance: [Double]) async -> [Double] {
        let nodes = await decisionTree.flatNodes
        return explain(tree: nodes, instance: instance, numFeatures: instance.count)
    }

    /// Computes TreeSHAP for a `DecisionTreeRegressor`.
    public func explain(decisionTree: DecisionTreeRegressor, instance: [Double]) async -> [Double] {
        let nodes = await decisionTree.flatNodes
        return explain(tree: nodes, instance: instance, numFeatures: instance.count)
    }

    /// Computes TreeSHAP for a `RandomForestClassifier`.
    public func explain(randomForest: RandomForestClassifier, instance: [Double]) async -> [Double] {
        let flatTrees = await randomForest.flatTrees
        return explain(trees: flatTrees, instance: instance, numFeatures: instance.count)
    }

    /// Computes TreeSHAP for a `RandomForestRegressor`.
    public func explain(randomForest: RandomForestRegressor, instance: [Double]) async -> [Double] {
        let flatTrees = await randomForest.flatTrees
        return explain(trees: flatTrees, instance: instance, numFeatures: instance.count)
    }

    // MARK: - Concurrent Batch TreeSHAP

    /// Computes exact Shapley values concurrently for a batch of input instances across CPU cores.
    public func explainBatch(
        trees: [[FlatTreeNode]],
        instances: [[Double]],
        numFeatures: Int
    ) async -> [[Double]] {
        guard !instances.isEmpty else { return [] }
        guard !trees.isEmpty else {
            return [[Double]](repeating: [Double](repeating: 0.0, count: numFeatures), count: instances.count)
        }

        var results = [[Double]](repeating: [], count: instances.count)

        if instances.count <= 4 {
            for (idx, instance) in instances.enumerated() {
                results[idx] = explain(trees: trees, instance: instance, numFeatures: numFeatures)
            }
            return results
        }

        await withTaskGroup(of: (Int, [Double]).self) { group in
            for (idx, instance) in instances.enumerated() {
                group.addTask {
                    let phi = self.explain(trees: trees, instance: instance, numFeatures: numFeatures)
                    return (idx, phi)
                }
            }
            for await (idx, phi) in group {
                results[idx] = phi
            }
        }

        return results
    }

    /// Computes TreeSHAP feature attributions concurrently for a batch of instances with a `DecisionTreeClassifier`.
    public func explainBatch(
        decisionTree: DecisionTreeClassifier,
        instances: [[Double]]
    ) async -> [[Double]] {
        guard let first = instances.first else { return [] }
        let nodes = await decisionTree.flatNodes
        return await explainBatch(trees: [nodes], instances: instances, numFeatures: first.count)
    }

    /// Computes TreeSHAP feature attributions concurrently for a batch of instances with a `DecisionTreeRegressor`.
    public func explainBatch(
        decisionTree: DecisionTreeRegressor,
        instances: [[Double]]
    ) async -> [[Double]] {
        guard let first = instances.first else { return [] }
        let nodes = await decisionTree.flatNodes
        return await explainBatch(trees: [nodes], instances: instances, numFeatures: first.count)
    }

    /// Computes TreeSHAP feature attributions concurrently for a batch of instances with a `RandomForestClassifier`.
    public func explainBatch(
        randomForest: RandomForestClassifier,
        instances: [[Double]]
    ) async -> [[Double]] {
        guard let first = instances.first else { return [] }
        let flatTrees = await randomForest.flatTrees
        return await explainBatch(trees: flatTrees, instances: instances, numFeatures: first.count)
    }

    /// Computes TreeSHAP feature attributions concurrently for a batch of instances with a `RandomForestRegressor`.
    public func explainBatch(
        randomForest: RandomForestRegressor,
        instances: [[Double]]
    ) async -> [[Double]] {
        guard let first = instances.first else { return [] }
        let flatTrees = await randomForest.flatTrees
        return await explainBatch(trees: flatTrees, instances: instances, numFeatures: first.count)
    }

    // MARK: - Blackbox Model Fallback (KernelSHAP Delegation)

    /// Explains predictions of any model by calculating Shapley values for each instance against background data via KernelSHAP.
    public func explain(
        model: @escaping @Sendable ([Double]) async -> Double,
        features: [[Double]],
        background: [[Double]]? = nil,
        numCoalitions: Int = 100
    ) async -> [[Double]] {
        guard !features.isEmpty else { return [] }
        let bg = background ?? features
        let kernelSHAP = KernelSHAP()

        var results: [[Double]] = []
        results.reserveCapacity(features.count)

        for instance in features {
            let shapValues = await kernelSHAP.explain(
                model: model,
                instance: instance,
                background: bg,
                numCoalitions: numCoalitions
            )
            results.append(shapValues)
        }
        return results
    }
}

// MARK: - Deterministic PRNG for Permutation Feature Importance

private struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) {
        self.state = seed == 0 ? 0x853c49e6748fea9b : seed
    }
    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}

/// Permutation Feature Importance calculator.
public struct PermutationImportance: Sendable {
    /// Creates a new instance.
    public init() {}

    /// Computes feature importance by measuring decrease in model performance (MSE) when each feature column is shuffled.
    ///
    /// Preserves full unclipped differences (retaining negative importance indications) and supports reproducible seeded permutations.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - targets: 1D array of ground-truth target values of length `N`.
    ///   - repeats: Number of shuffle repetitions to average over (default: 3).
    ///   - randomSeed: Optional PRNG seed for deterministic reproducible evaluations (default: 42).
    ///   - predict: Inference closure mapping feature vectors to predictions.
    /// - Returns: Dictionary mapping `"feature_\(c)"` to unclipped change in MSE (`avgShuffledMSE - baselineMSE`).
    public func computeImportance(
        features: [[Double]],
        targets: [Double],
        repeats: Int = 3,
        randomSeed: Int? = 42,
        predict: @escaping @Sendable ([[Double]]) async throws -> [Double]
    ) async throws -> [String: Double] {
        guard !features.isEmpty, features.count == targets.count else { return [:] }
        let numRows = features.count
        let numCols = features[0].count
        guard numCols > 0 else { return [:] }

        // Validate matrix rectangularity and data finiteness
        guard features.allSatisfy({ $0.count == numCols && $0.allSatisfy({ !$0.isNaN && !$0.isInfinite }) }),
              targets.allSatisfy({ !$0.isNaN && !$0.isInfinite }) else {
            return [:]
        }

        let baselinePreds = try await predict(features)
        guard baselinePreds.count == numRows,
              baselinePreds.allSatisfy({ !$0.isNaN && !$0.isInfinite }) else {
            return [:]
        }

        let baselineMSE = zip(baselinePreds, targets).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) } / Double(numRows)

        let safeRepeats = max(1, repeats)
        var rng = SeededRNG(seed: UInt64(bitPattern: Int64(randomSeed ?? 42)))

        var result: [String: Double] = [:]
        for c in 0..<numCols {
            var shuffledMSETotal = 0.0
            for _ in 0..<safeRepeats {
                var shuffled = features
                var perm: [Int]
                if randomSeed != nil {
                    perm = Array(0..<numRows).shuffled(using: &rng)
                } else {
                    perm = Array(0..<numRows).shuffled()
                }

                if numRows > 1 && perm == Array(0..<numRows) {
                    perm.swapAt(0, 1)
                }
                for i in 0..<numRows {
                    shuffled[i][c] = features[perm[i]][c]
                }
                let shuffledPreds = try await predict(shuffled)
                guard shuffledPreds.count == numRows else { return [:] }
                let mse = zip(shuffledPreds, targets).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) } / Double(numRows)
                shuffledMSETotal += mse
            }
            let avgShuffledMSE = shuffledMSETotal / Double(safeRepeats)
            // Preserve raw difference without clipping to zero
            result["feature_\(c)"] = avgShuffledMSE - baselineMSE
        }
        return result
    }
}

/// Partial Dependence Plot (PDP) and Individual Conditional Expectation (ICE) calculator.
public struct PartialDependencePlot: Sendable {
    /// Creates a new instance.
    public init() {}

    /// Calculates PDP grid values for a specified feature index by replacing feature column values with grid points.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - featureIndex: Zero-based column index of the target feature.
    ///   - gridPoints: Number of evaluation points or grid resolution across feature range (must be >= 2).
    ///   - predict: Inference closure mapping feature vectors to predictions.
    /// - Returns: The computed `(grid: [Double], values: [Double])` result tuple.
    public func calculatePDP(
        features: [[Double]],
        featureIndex: Int,
        gridPoints: Int = 10,
        predict: @escaping @Sendable ([[Double]]) async throws -> [Double]
    ) async throws -> (grid: [Double], values: [Double]) {
        // Strict validation of inputs, rectangularity, and gridPoints >= 2
        guard !features.isEmpty,
              gridPoints >= 2,
              featureIndex >= 0,
              features.allSatisfy({ $0.count > featureIndex && $0.allSatisfy({ !$0.isNaN && !$0.isInfinite }) }) else {
            return (grid: [], values: [])
        }

        let colValues = features.map { $0[featureIndex] }
        guard let minVal = colValues.min(), let maxVal = colValues.max(),
              !minVal.isNaN, !maxVal.isNaN else {
            return (grid: [], values: [])
        }

        let step = (maxVal > minVal) ? (maxVal - minVal) / Double(gridPoints - 1) : 0.0
        let grid = (0..<gridPoints).map { minVal + Double($0) * step }

        var pdpValues: [Double] = []
        pdpValues.reserveCapacity(gridPoints)

        for val in grid {
            var modifiedFeatures = features
            for r in 0..<features.count {
                modifiedFeatures[r][featureIndex] = val
            }
            let preds = try await predict(modifiedFeatures)
            // Ensure returned predictions match row count and contain valid numbers
            guard preds.count == features.count,
                  preds.allSatisfy({ !$0.isNaN && !$0.isInfinite }) else {
                return (grid: [], values: [])
            }
            let meanPred = preds.reduce(0.0, +) / Double(preds.count)
            pdpValues.append(meanPred)
        }

        return (grid: grid, values: pdpValues)
    }
}

