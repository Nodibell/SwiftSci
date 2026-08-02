import Foundation
import SwiftML

/// Performs parallel Randomized Search over random hyperparameter combinations
/// via K-Fold Cross-Validation.
public struct RandomizedSearchCV: Sendable {

    /// Evaluation result payload for decision tree parameter combinations.
    public struct Result: Sendable, Comparable, Equatable {
        /// Evaluated maximum tree depth hyperparameter.
        public let maxDepth: Int
        /// Evaluated node split criterion hyperparameter.
        public let criterion: SplitCriterion
        /// Cross-validated mean evaluation metric score across all folds.
        public let meanScore: Double
        /// Standard deviation of evaluation scores across folds.
        public let stdScore: Double

        /// Initializes a decision tree search result.
        /// - Parameters:
        ///   - maxDepth: Evaluated maximum tree depth.
        ///   - criterion: Evaluated node split criterion.
        ///   - meanScore: Cross-validated mean evaluation metric score.
        ///   - stdScore: Standard deviation of evaluation scores.
        public init(maxDepth: Int, criterion: SplitCriterion, meanScore: Double, stdScore: Double) {
            self.maxDepth = maxDepth
            self.criterion = criterion
            self.meanScore = meanScore
            self.stdScore = stdScore
        }

        public static func < (lhs: Result, rhs: Result) -> Bool {
            lhs.meanScore < rhs.meanScore
        }
        
        public static func == (lhs: Result, rhs: Result) -> Bool {
            lhs.maxDepth == rhs.maxDepth && lhs.criterion == rhs.criterion && lhs.meanScore == rhs.meanScore && lhs.stdScore == rhs.stdScore
        }
    }

    /// Generic result payload holding arbitrary parameter combinations.
    public struct GenericResult<Params: Sendable>: Sendable, Comparable, Equatable {
        /// Sampled hyperparameter combination instance.
        public let params: Params
        /// Cross-validated mean evaluation metric score across all folds.
        public let meanScore: Double
        /// Standard deviation of evaluation scores across folds.
        public let stdScore: Double

        /// Initializes a generic randomized search result payload.
        /// - Parameters:
        ///   - params: Sampled hyperparameter combination.
        ///   - meanScore: Cross-validated mean metric score.
        ///   - stdScore: Standard deviation of evaluation scores.
        public init(params: Params, meanScore: Double, stdScore: Double) {
            self.params = params
            self.meanScore = meanScore
            self.stdScore = stdScore
        }

        public static func < (lhs: GenericResult<Params>, rhs: GenericResult<Params>) -> Bool {
            lhs.meanScore < rhs.meanScore
        }
        
        public static func == (lhs: GenericResult<Params>, rhs: GenericResult<Params>) -> Bool {
            lhs.meanScore == rhs.meanScore && lhs.stdScore == rhs.stdScore
        }
    }

    /// The max depth values.
    public let maxDepthValues: [Int]
    /// The criterion values.
    public let criterionValues: [SplitCriterion]
    /// The n iter.
    public let nIter: Int
    /// The n splits.
    public let nSplits: Int
    /// The seed.
    public let seed: Int

    /// Creates a new instance.
    /// - Parameters:
    ///   - maxDepthValues: The max depth values.
    ///   - criterionValues: The criterion values.
    ///   - nIter: The n iter.
    ///   - nSplits: The n splits.
    ///   - seed: The seed.
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

    /// Runs randomized search over generic parameter candidates and custom estimator factory.
    public func searchGeneric<P: Sendable, E: ClassifierEstimator>(
        candidates: [P],
        features: [[Double]],
        targets: [Double],
        estimatorBuilder: @escaping @Sendable (P) -> E
    ) async throws -> [GenericResult<P>] {
        guard !candidates.isEmpty else { return [] }
        let count = min(nIter, candidates.count)
        let sampled = Array(candidates.prefix(count))
        let nSplits = self.nSplits
        let cvSeed = self.seed

        let results: [GenericResult<P>] = try await withThrowingTaskGroup(of: GenericResult<P>.self) { group in
            for p in sampled {
                group.addTask {
                    let factory: @Sendable () -> E = { estimatorBuilder(p) }
                    let cvResult = try await CrossValidator.crossValidate(
                        factory,
                        features: features,
                        targets: targets,
                        nSplits: nSplits,
                        seed: cvSeed
                    )
                    return GenericResult(params: p, meanScore: cvResult.mean, stdScore: cvResult.std)
                }
            }
            var allResults = [GenericResult<P>]()
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
