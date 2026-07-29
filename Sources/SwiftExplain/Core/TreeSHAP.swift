import Foundation
import SwiftML

/// SHAP explainer delegating to KernelSHAP model interpretations.
public struct TreeSHAP: Sendable {
    /// Creates a new instance.
    public init() {}

    /// Explains predictions of a model by calculating Shapley values for each instance against background data via KernelSHAP.
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
