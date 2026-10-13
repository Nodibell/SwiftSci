import Foundation
import SwiftML

// MARK: - GridSearchCV

/// Performs parallel Grid Search over a hyperparameter grid for DecisionTreeClassifier
/// to find the best hyperparameter combination via Stratified K-Fold Cross-Validation.
public struct GridSearchCV: Sendable {

    /// Single hyperparameter trial evaluation result payload from Grid Search.
    public struct Result: Sendable, Comparable, Equatable {
        /// Evaluated maximum tree depth hyperparameter value.
        public let maxDepth: Int
        /// Evaluated node split criterion hyperparameter value.
        public let criterion: SplitCriterion
        /// Cross-validated mean evaluation metric score across all folds.
        public let meanScore: Double
        /// Standard deviation of evaluation scores across folds.
        public let stdScore: Double

        /// Initializes a decision tree grid search result.
        public init(maxDepth: Int, criterion: SplitCriterion, meanScore: Double, stdScore: Double) {
            self.maxDepth = maxDepth
            self.criterion = criterion
            self.meanScore = meanScore
            self.stdScore = stdScore
        }

        public static func < (lhs: Result, rhs: Result) -> Bool {
            if lhs.meanScore != rhs.meanScore {
                return lhs.meanScore < rhs.meanScore
            }
            if lhs.stdScore != rhs.stdScore {
                // Lower standard deviation indicates higher stability across folds (better)
                return lhs.stdScore > rhs.stdScore
            }
            if lhs.maxDepth != rhs.maxDepth {
                // Simpler / shallower models are preferred in case of score ties
                return lhs.maxDepth > rhs.maxDepth
            }
            return lhs.criterionOrder < rhs.criterionOrder
        }

        public static func == (lhs: Result, rhs: Result) -> Bool {
            lhs.maxDepth == rhs.maxDepth && lhs.criterion == rhs.criterion && lhs.meanScore == rhs.meanScore && lhs.stdScore == rhs.stdScore
        }

        private var criterionOrder: Int {
            switch criterion {
            case .gini: return 0
            case .entropy: return 1
            case .mse: return 2
            }
        }
    }

    /// Candidate maximum depth values evaluated during search.
    public let maxDepthValues: [Int]
    /// Candidate split criterion values evaluated during search.
    public let criterionValues: [SplitCriterion]
    /// Number of cross-validation folds.
    public let nSplits: Int
    /// Random seed for fold generation.
    public let seed: Int

    /// Creates a new Grid Search cross-validator for DecisionTreeClassifier.
    /// - Parameters:
    ///   - maxDepthValues: Candidate tree depths (must be non-empty, positive integers).
    ///   - criterionValues: Candidate split criteria (must be non-empty).
    ///   - nSplits: Number of cross-validation folds (must be at least 2).
    ///   - seed: Random seed.
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
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - targets: 1D array of ground-truth target values of length `N`.
    /// - Throws: `ValidationError` or `SwiftMLError` if parameter grids are empty, folds are invalid, or evaluations fail.
    /// - Returns: The computed [Result] result instance.
    public func search(features: [[Double]], targets: [Double]) async throws -> [Result] {
        guard nSplits >= 2 else {
            throw ValidationError.invalidFoldCount(nSplits)
        }
        guard !maxDepthValues.isEmpty else {
            throw ValidationError.invalidParameter("maxDepthValues must not be empty")
        }
        guard !criterionValues.isEmpty else {
            throw ValidationError.invalidParameter("criterionValues must not be empty")
        }
        for d in maxDepthValues {
            guard d > 0 else {
                throw ValidationError.invalidParameter("maxDepthValues must contain positive integers, got \(d)")
            }
        }
        guard !features.isEmpty, !targets.isEmpty else {
            throw ValidationError.emptyDataset
        }
        guard features.count == targets.count else {
            throw ValidationError.dimensionMismatch(features: features.count, targets: targets.count)
        }
        guard features.count >= nSplits else {
            throw ValidationError.insufficientSamples(samples: features.count, required: nSplits)
        }

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
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - targets: 1D array of ground-truth target values of length `N`.
    /// - Throws: `ValidationError` or `SwiftMLError` if parameter grids are empty, folds are invalid, or evaluations fail.
    /// - Returns: The computed Result? result instance.
    public func bestParams(features: [[Double]], targets: [Double]) async throws -> Result? {
        let results = try await search(features: features, targets: targets)
        return results.first
    }
}
