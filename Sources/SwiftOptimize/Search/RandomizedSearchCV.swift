import Foundation
import SwiftML

/// Performs parallel Randomized Search over random hyperparameter combinations
/// via K-Fold Cross-Validation.
public struct RandomizedSearchCV: Sendable {

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
    public let nIter: Int
    public let nSplits: Int
    public let seed: Int

    public init(
        maxDepthValues: [Int] = [3, 5, 7, 10, 15, 20],
        criterionValues: [SplitCriterion] = [.gini, .entropy],
        nIter: Int = 5,
        nSplits: Int = 5,
        seed: Int = 42
    ) {
        self.maxDepthValues = maxDepthValues
        self.criterionValues = criterionValues
        self.nIter = nIter
        self.nSplits = nSplits
        self.seed = seed
    }

    /// Runs randomized search over randomly sampled parameter combinations.
    public func search(features: [[Double]], targets: [Double]) async throws -> [Result] {
        let allPairs: [(Int, SplitCriterion)] = maxDepthValues.flatMap { d in criterionValues.map { c in (d, c) } }
        guard !allPairs.isEmpty else { return [] }

        var rng = SeededRandom(seed: seed)
        var sampledPairs = [(Int, SplitCriterion)]()
        let count = min(nIter, allPairs.count)
        
        var available = allPairs
        for _ in 0..<count {
            let idx = rng.nextInt(upperBound: available.count)
            sampledPairs.append(available.remove(at: idx))
        }

        let nSplits = self.nSplits
        let cvSeed = self.seed

        let results: [Result] = try await withThrowingTaskGroup(of: Result.self) { group in
            for (depth, criterion) in sampledPairs {
                group.addTask {
                    let cvResult = try await CrossValidator.crossValidate(
                        classifier: (maxDepth: depth, criterion: criterion),
                        features: features,
                        targets: targets,
                        nSplits: nSplits,
                        seed: cvSeed
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
