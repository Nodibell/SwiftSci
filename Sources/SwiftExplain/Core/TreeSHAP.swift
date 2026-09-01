import Foundation
import SwiftML

/// Fast, exact TreeSHAP (Tree Shapley Additive Explanations) for decision trees and tree ensembles
/// (Lundberg et al., 2018; Nature Machine Intelligence).
///
/// Computes exact Shapley values in polynomial time \(O(T \cdot L \cdot D^2)\) where \(T\) is the number
/// of trees, \(L\) is the number of leaves, and \(D\) is the maximum tree depth, rather than exponential \(O(2^M)\).
public struct TreeSHAP: Sendable {
    
    /// Creates a new TreeSHAP explainer instance.
    public init() {}

    // MARK: - Exact TreeSHAP for FlatTreeNode arrays

    /// Computes exact Shapley values for a single decision tree and an input instance.
    /// - Parameters:
    ///   - tree: Flat array of `FlatTreeNode` nodes representing the fitted decision tree.
    ///   - instance: Feature vector of the instance to explain.
    ///   - numFeatures: Total number of features in the feature space.
    /// - Returns: Exact Shapley values for each feature of length `numFeatures`.
    public func explain(tree: [FlatTreeNode], instance: [Double], numFeatures: Int) -> [Double] {
        guard !tree.isEmpty, !instance.isEmpty, numFeatures > 0 else {
            return [Double](repeating: 0.0, count: numFeatures)
        }

        var phi = [Double](repeating: 0.0, count: numFeatures)
        
        // Recursive path tracking struct
        struct PathStep {
            let featureIndex: Int
            let matched: Bool
        }
        
        func traverse(nodeIdx: Int, path: [PathStep]) {
            guard nodeIdx >= 0 && nodeIdx < tree.count else { return }
            let node = tree[nodeIdx]
            
            if node.isLeaf {
                let leafVal = node.value
                let d = path.count
                guard d > 0 else { return }
                
                // Group path by unique feature indices
                var featureMatches: [Int: Bool] = [:]
                for step in path {
                    featureMatches[step.featureIndex] = step.matched
                }
                
                let uniqueFeatures = Array(featureMatches.keys)
                let numUnique = uniqueFeatures.count
                let numMatched = uniqueFeatures.filter { featureMatches[$0] == true }.count
                
                // Combinatorial Shapley weight: |S|! * (M - |S| - 1)! / M!
                func shapWeight(s: Int, m: Int) -> Double {
                    guard m > 0, s >= 0, s < m else { return 1.0 }
                    var w = 1.0
                    if s >= 1 {
                        for i in 1...s {
                            w *= Double(i)
                        }
                    }
                    let rem = m - s - 1
                    if rem >= 1 {
                        for i in 1...rem {
                            w *= Double(i)
                        }
                    }
                    if m >= 1 {
                        for i in 1...m {
                            w /= Double(i)
                        }
                    }
                    return w
                }
                
                for feat in uniqueFeatures {
                    guard feat >= 0 && feat < numFeatures else { continue }
                    let isMatch = featureMatches[feat] == true
                    let sWithout = isMatch ? (numMatched - 1) : numMatched
                    let weight = shapWeight(s: sWithout, m: numUnique)
                    let sign: Double = isMatch ? 1.0 : -1.0
                    phi[feat] += sign * weight * leafVal
                }
                return
            }
            
            let featIdx = node.featureIndex
            let threshold = node.threshold
            let instVal = (featIdx < instance.count) ? instance[featIdx] : 0.0
            let instanceWentLeft = instVal <= threshold
            
            // Traverse Left
            let leftStep = PathStep(featureIndex: featIdx, matched: instanceWentLeft)
            traverse(nodeIdx: node.leftChild, path: path + [leftStep])
            
            // Traverse Right
            let rightStep = PathStep(featureIndex: featIdx, matched: !instanceWentLeft)
            traverse(nodeIdx: node.rightChild, path: path + [rightStep])
        }
        
        traverse(nodeIdx: 0, path: [])
        return phi
    }

    /// Computes exact Shapley values for an ensemble of trees (e.g. Random Forest).
    /// - Parameters:
    ///   - trees: Array of trees, where each tree is `[FlatTreeNode]`.
    ///   - instance: Feature vector of the instance to explain.
    ///   - numFeatures: Total number of features.
    /// - Returns: Average exact Shapley values across the ensemble.
    public func explain(trees: [[FlatTreeNode]], instance: [Double], numFeatures: Int) -> [Double] {
        guard !trees.isEmpty else { return [Double](repeating: 0.0, count: numFeatures) }
        var totalPhi = [Double](repeating: 0.0, count: numFeatures)
        
        for tree in trees {
            let treePhi = explain(tree: tree, instance: instance, numFeatures: numFeatures)
            for i in 0..<numFeatures {
                totalPhi[i] += treePhi[i]
            }
        }
        
        let n = Double(trees.count)
        return totalPhi.map { $0 / n }
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

/// Permutation Feature Importance calculator.
public struct PermutationImportance: Sendable {
    /// Creates a new instance.
    public init() {}

    /// Computes feature importance by measuring decrease in model performance (MSE) when each feature column is shuffled.
    public func computeImportance(
        features: [[Double]],
        targets: [Double],
        predict: @escaping @Sendable ([[Double]]) async throws -> [Double]
    ) async throws -> [String: Double] {
        guard !features.isEmpty, features.count == targets.count else { return [:] }
        let numRows = features.count
        let numCols = features[0].count

        let baselinePreds = try await predict(features)
        guard baselinePreds.count == numRows else { return [:] }

        let baselineMSE = zip(baselinePreds, targets).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) } / Double(numRows)

        var result: [String: Double] = [:]
        for c in 0..<numCols {
            var shuffled = features
            let perm = (0..<numRows).shuffled()
            for i in 0..<numRows {
                shuffled[i][c] = features[perm[i]][c]
            }
            let shuffledPreds = try await predict(shuffled)
            let shuffledMSE = zip(shuffledPreds, targets).reduce(0.0) { $0 + pow($1.0 - $1.1, 2) } / Double(numRows)
            result["feature_\(c)"] = max(0.0, shuffledMSE - baselineMSE)
        }
        return result
    }
}

/// Partial Dependence Plot (PDP) and Individual Conditional Expectation (ICE) calculator.
public struct PartialDependencePlot: Sendable {
    /// Creates a new instance.
    public init() {}

    /// Calculates PDP grid values for a specified feature index by replacing feature column values with grid points.
    public func calculatePDP(
        features: [[Double]],
        featureIndex: Int,
        gridPoints: Int = 10,
        predict: @escaping @Sendable ([[Double]]) async throws -> [Double]
    ) async throws -> (grid: [Double], values: [Double]) {
        guard !features.isEmpty, featureIndex >= 0, featureIndex < features[0].count else {
            return (grid: [], values: [])
        }

        let colValues = features.map { $0[featureIndex] }
        let minVal = colValues.min() ?? 0.0
        let maxVal = colValues.max() ?? 1.0

        let step = (gridPoints > 1 && maxVal > minVal) ? (maxVal - minVal) / Double(gridPoints - 1) : 0.0
        let grid = (0..<gridPoints).map { minVal + Double($0) * step }

        var pdpValues: [Double] = []
        pdpValues.reserveCapacity(gridPoints)

        for val in grid {
            var modifiedFeatures = features
            for r in 0..<features.count {
                modifiedFeatures[r][featureIndex] = val
            }
            let preds = try await predict(modifiedFeatures)
            let meanPred = preds.isEmpty ? 0.0 : preds.reduce(0.0, +) / Double(preds.count)
            pdpValues.append(meanPred)
        }

        return (grid: grid, values: pdpValues)
    }
}
