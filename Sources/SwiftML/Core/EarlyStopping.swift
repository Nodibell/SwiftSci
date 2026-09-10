import Foundation

/// Metric optimization direction monitored by early stopping.
public enum MetricDirection: Sendable, Equatable {
    /// Desired metric decreases (e.g. loss, mean squared error, cross-entropy).
    case minimize
    /// Desired metric increases (e.g. accuracy, F1-score, R2, ROC-AUC).
    case maximize
}

/// A training callback that halts iterative model fitting when validation performance ceases to improve.
///
/// Early stopping tracks a validation metric across successive training epochs or boosting iterations.
/// If the metric fails to improve by at least `minDelta` for `patience` consecutive evaluations,
/// training terminates prematurely to prevent overfitting and avoid unnecessary computation.
///
/// ## Thread Safety
/// Conforms to `Sendable` and value semantics. Safe to pass across asynchronous actor boundaries.
public struct EarlyStopping: Sendable {

    /// Number of consecutive epochs with no improvement before stopping.
    public let patience: Int

    /// Minimum absolute change in the monitored quantity to qualify as an improvement.
    public let minDelta: Double

    /// Direction of optimization (`.minimize` for losses, `.maximize` for scores).
    public let direction: MetricDirection

    /// When true, restores model parameters to the epoch with the best validation score upon termination.
    public let restoreBestWeights: Bool

    /// Best score observed so far (nil before first evaluation).
    public private(set) var bestScore: Double?

    /// Number of epochs elapsed since the last improvement.
    public private(set) var wait: Int = 0

    /// Indicates whether training should halt immediately.
    public private(set) var shouldStop: Bool = false

    /// Epoch index corresponding to `bestScore`.
    public private(set) var bestEpoch: Int = 0

    /// Creates a new early stopping callback instance.
    ///
    /// - Parameters:
    ///   - patience: Number of consecutive epochs with no improvement before stopping. Default is `10`.
    ///   - minDelta: Minimum absolute change in the monitored quantity to qualify as an improvement. Default is `1e-4`.
    ///   - direction: Whether to minimize or maximize the monitored metric. Default is `.minimize`.
    ///   - restoreBestWeights: When true, restores model parameters to the epoch with the best validation score. Default is `true`.
    ///
    /// ## Thread Safety
    /// Conforms to `Sendable`.
    ///
    /// ## Complexity
    /// \(O(1)\) initialization.
    public init(
        patience: Int = 10,
        minDelta: Double = 1e-4,
        direction: MetricDirection = .minimize,
        restoreBestWeights: Bool = true
    ) {
        precondition(patience > 0, "patience must be > 0")
        precondition(minDelta >= 0.0, "minDelta must be >= 0")

        self.patience = patience
        self.minDelta = minDelta
        self.direction = direction
        self.restoreBestWeights = restoreBestWeights
    }

    /// Evaluates the current epoch's metric value and updates early stopping state.
    ///
    /// - Parameters:
    ///   - currentScore: The evaluated validation metric for the current epoch.
    ///   - epoch: Current epoch index (0-indexed).
    /// - Returns: A tuple `(shouldStop: Bool, isBest: Bool)` indicating whether to stop and if this epoch is the best.
    ///
    /// ## Thread Safety
    /// Mutating method on value type; thread-safe when isolated.
    ///
    /// ## Complexity
    /// \(O(1)\).
    @discardableResult
    public mutating func step(currentScore: Double, epoch: Int) -> (shouldStop: Bool, isBest: Bool) {
        guard let best = bestScore else {
            bestScore = currentScore
            bestEpoch = epoch
            wait = 0
            shouldStop = false
            return (shouldStop: false, isBest: true)
        }

        let isImproved: Bool
        switch direction {
        case .minimize:
            isImproved = (best - currentScore) > minDelta
        case .maximize:
            isImproved = (currentScore - best) > minDelta
        }

        if isImproved {
            bestScore = currentScore
            bestEpoch = epoch
            wait = 0
            return (shouldStop: false, isBest: true)
        } else {
            wait += 1
            if wait >= patience {
                shouldStop = true
            }
            return (shouldStop: shouldStop, isBest: false)
        }
    }

    /// Resets the internal state of the early stopping tracker.
    ///
    /// ## Complexity
    /// \(O(1)\).
    public mutating func reset() {
        bestScore = nil
        wait = 0
        shouldStop = false
        bestEpoch = 0
    }
}
