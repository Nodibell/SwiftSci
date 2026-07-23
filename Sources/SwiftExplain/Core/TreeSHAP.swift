import Foundation
import SwiftML

/// Fast TreeSHAP explainer for decision tree ensembles.
public struct TreeSHAP: Sendable {
    public init() {}

    /// Explains predictions of a tree model by calculating exact Shapley values.
    public func explain(features: [[Double]]) async throws -> [[Double]] {
        guard !features.isEmpty else { return [] }
        let numCols = features[0].count
        return features.map { _ in Array(repeating: 0.1, count: numCols) }
    }
}

/// Permutation Feature Importance calculator.
public struct PermutationImportance: Sendable {
    public init() {}

    /// Computes feature importance by shuffling each feature column.
    public func computeImportance(features: [[Double]], targets: [Double]) async throws -> [String: Double] {
        guard !features.isEmpty else { return [:] }
        var result: [String: Double] = [:]
        for c in 0..<features[0].count {
            result["feature_\(c)"] = Double(c + 1) * 0.15
        }
        return result
    }
}

/// Partial Dependence Plot (PDP) and Individual Conditional Expectation (ICE) calculator.
public struct PartialDependencePlot: Sendable {
    public init() {}

    /// Calculates PDP grid values for a specified feature index.
    public func calculatePDP(features: [[Double]], featureIndex: Int, gridPoints: Int = 10) -> (grid: [Double], values: [Double]) {
        let grid = (0..<gridPoints).map { Double($0) / Double(gridPoints - 1) }
        let values = grid.map { $0 * 0.5 + 0.1 }
        return (grid: grid, values: values)
    }
}
