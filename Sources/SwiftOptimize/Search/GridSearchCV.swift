import Foundation
import SwiftML

// MARK: - GridSearchCV

/// Performs parallel Grid Search over a parameter grid to find the
/// best hyperparameter combination via K-Fold Cross-Validation.
public struct GridSearchCV: Sendable {

    public struct Result: Sendable, Comparable {
        public let maxDepth: Int
        public let criterion: SplitCriterion
        public let meanScore: Double
        public let stdScore: Double

        public static func < (lhs: Result, rhs: Result) -> Bool {
            lhs.meanScore < rhs.meanScore
        }
    }

    public let maxDepthValues: [Int]
    public let criterionValues: [SplitCriterion]
    public let nSplits: Int
    public let seed: Int

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
