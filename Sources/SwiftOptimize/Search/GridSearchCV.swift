import Foundation
import SwiftML

// MARK: - GridSearchCV

/// Performs parallel Grid Search over a parameter grid to find the
/// best hyperparameter combination via K-Fold Cross-Validation.
public struct GridSearchCV: Sendable {

    /// Single hyperparameter trial evaluation result payload from Grid Search.
    public struct Result: Sendable, Comparable {
        /// Evaluated maximum tree depth hyperparameter value.
        public let maxDepth: Int
        /// Evaluated node split criterion hyperparameter value.
        public let criterion: SplitCriterion
        /// Cross-validated mean evaluation metric score across all folds.
        public let meanScore: Double
        /// Standard deviation of evaluation scores across folds.
        public let stdScore: Double

        public static func < (lhs: Result, rhs: Result) -> Bool {
            lhs.meanScore < rhs.meanScore
        }
    }

    /// The max depth values.
    public let maxDepthValues: [Int]
    /// The criterion values.
    public let criterionValues: [SplitCriterion]
    /// The n splits.
    public let nSplits: Int
    /// The seed.
    public let seed: Int

    /// Creates a new instance.
    /// - Parameters:
    ///   - maxDepthValues: The max depth values.
    ///   - criterionValues: The criterion values.
    ///   - nSplits: The n splits.
    ///   - seed: The seed.
    public init(
        maxDepthValues: [Int] = [3, 5, 7, 10],
        criterionValues: [SplitCriterion] = [.gini, .entropy],
        nSplits: Int = 5,
        seed: Int = 42
    ) {
        self.maxDepthValues = maxDepthValues
        self.criterionValues = criterionValues
        self.nSplits = nSplits
        self.seed = seed
    }

    /// Runs grid search over DecisionTreeClassifier and returns all results sorted best-first.
    public func search(features: [[Double]], targets: [Double]) async throws -> [Result] {
        let nSplits = self.nSplits
        let seed = self.seed
        let pairs: [(Int, SplitCriterion)] = maxDepthValues.flatMap { d in criterionValues.map { c in (d, c) } }

        let results: [Result] = try await withThrowingTaskGroup(of: Result.self) { group in
            for (depth, criterion) in pairs {
                group.addTask {
                    let cvResult = try await CrossValidator.crossValidate(
                        classifier: (maxDepth: depth, criterion: criterion),
                        features: features,
                        targets: targets,
                        nSplits: nSplits,
                        seed: seed
                    )
                    return Result(maxDepth: depth, criterion: criterion,
                                  meanScore: cvResult.mean, stdScore: cvResult.std)
                }
            }
            var allResults = [Result]()
            for try await r in group { allResults.append(r) }
            return allResults
        }

        return results.sorted().reversed()
    }

    /// Convenience: returns only the best parameter combination.
    public func bestParams(features: [[Double]], targets: [Double]) async throws -> Result? {
        let results = try await search(features: features, targets: targets)
        return results.first
    }
}
