import Foundation

// MARK: - Platt Scaling

/// Calibrates uncalibrated raw scores / logits into probabilities using sigmoid (Platt) scaling.
/// Fits parameters `a` and `b` in `P(y=1|f) = sigmoid(a * f + b)` via gradient descent on log-loss.
public final class PlattScaling: @unchecked Sendable {
    /// The slope parameter `a`.
    public private(set) var a: Double = 1.0
    /// The intercept parameter `b`.
    public private(set) var b: Double = 0.0
    /// Whether the calibrator has been fitted.
    public private(set) var isFitted: Bool = false

    /// Creates a new instance.
    public init() {}

    /// Fits Platt scaling parameters `a` and `b` using score and binary target pairs.
    /// - Parameters:
    ///   - scores: Uncalibrated decision scores or logits.
    ///   - targets: Binary target labels (`0` or `1`).
    ///   - maxIterations: Maximum number of gradient descent iterations.
    ///   - lr: Gradient descent learning rate.
    public func fit(scores: [Double], targets: [Int], maxIterations: Int = 100, lr: Double = 0.05) {
        guard !scores.isEmpty, scores.count == targets.count else { return }

        // Gradient descent optimization for log loss parameters a and b
        var currA = 1.0
        var currB = 0.0

        for _ in 0..<maxIterations {
            var gradA = 0.0
            var gradB = 0.0

            for (s, t) in zip(scores, targets) {
                let logit = currA * s + currB
                let p = sigmoid(logit)
                let err = p - Double(t)
                gradA += err * s
                gradB += err
            }

            let n = Double(scores.count)
            currA -= lr * (gradA / n)
            currB -= lr * (gradB / n)
        }

        self.a = currA
        self.b = currB
        self.isFitted = true
    }

    /// Predicts the calibrated positive-class probability for a single score.
    /// - Parameters:
    ///   - score: Uncalibrated decision score or logit.
    /// - Returns: Calibrated probability in `(0, 1)`.
    public func predictProbability(score: Double) -> Double {
        guard isFitted else { return sigmoid(score) }
        let logit = a * score + b
        return sigmoid(logit)
    }

    /// Predict probabilities.
    /// - Parameters:
    ///   - scores: The scores.
    /// - Returns: A `[Double]` result.
    public func predictProbabilities(scores: [Double]) -> [Double] {
        return scores.map { predictProbability(score: $0) }
    }
}

// MARK: - Isotonic Regression Calibration

/// Non-parametric probability calibration using the Pool Adjacent Violators Algorithm (PAVA).
public final class IsotonicRegression: @unchecked Sendable {
    private var thresholds: [Double] = []
    private var probabilities: [Double] = []
    /// is fitted.
    public private(set) var isFitted: Bool = false

    /// Creates a new instance.
    public init() {}

    /// Fit.
    /// - Parameters:
    ///   - scores: The scores.
    ///   - targets: The targets.
    public func fit(scores: [Double], targets: [Int]) {
        guard !scores.isEmpty, scores.count == targets.count else { return }

        // Sort by score ascending
        let paired = zip(scores, targets.map { Double($0) }).sorted { $0.0 < $1.0 }

        // PAVA algorithm
        var blocks: [(weight: Double, sum: Double, mean: Double, minScore: Double, maxScore: Double)] = []

        for (s, t) in paired {
            var newBlock = (weight: 1.0, sum: t, mean: t, minScore: s, maxScore: s)

            while let last = blocks.last, last.mean >= newBlock.mean {
                blocks.removeLast()
                let combinedWeight = last.weight + newBlock.weight
                let combinedSum = last.sum + newBlock.sum
                newBlock = (
                    weight: combinedWeight,
                    sum: combinedSum,
                    mean: combinedSum / combinedWeight,
                    minScore: last.minScore,
                    maxScore: newBlock.maxScore
                )
            }
            blocks.append(newBlock)
        }

        self.thresholds = blocks.map { $0.maxScore }
        self.probabilities = blocks.map { max(0.0, min(1.0, $0.mean)) }
        self.isFitted = true
    }

    /// Predict probability.
    /// - Parameters:
    ///   - score: The score.
    /// - Returns: A `Double` result.
    public func predictProbability(score: Double) -> Double {
        guard isFitted, !thresholds.isEmpty else { return 0.5 }
        if score <= thresholds[0] { return probabilities[0] }
        if score >= thresholds[thresholds.count - 1] { return probabilities[probabilities.count - 1] }

        // Linear interpolation between breakpoints
        for i in 0..<(thresholds.count - 1) {
            if score >= thresholds[i] && score <= thresholds[i + 1] {
                let t1 = thresholds[i]
                let t2 = thresholds[i + 1]
                let p1 = probabilities[i]
                let p2 = probabilities[i + 1]
                if t2 == t1 { return p1 }
                let alpha = (score - t1) / (t2 - t1)
                return p1 + alpha * (p2 - p1)
            }
        }
        return probabilities.last ?? 0.5
    }

    /// Predict probabilities.
    /// - Parameters:
    ///   - scores: The scores.
    /// - Returns: A `[Double]` result.
    public func predictProbabilities(scores: [Double]) -> [Double] {
        return scores.map { predictProbability(score: $0) }
    }
}
