import Foundation

// MARK: - Platt Scaling

/// Calibrates uncalibrated raw scores / logits into probabilities using sigmoid (Platt) scaling.
/// Fits parameters A and B in P(y=1|f) = 1 / (1 + exp(A * f + B)) via logistic OLS/Newton optimization.
public final class PlattScaling: @unchecked Sendable {
    /// The a.
    public private(set) var a: Double = -1.0
    /// The b.
    public private(set) var b: Double = 0.0
    /// is fitted.
    public private(set) var isFitted: Bool = false

    /// Creates a new instance.
    public init() {}

    /// Fit.
    /// - Parameters:
    ///   - scores: The scores.
    ///   - targets: The targets.
    ///   - maxIterations: The max iterations.
    ///   - lr: The lr.
    public func fit(scores: [Double], targets: [Int], maxIterations: Int = 100, lr: Double = 0.05) {
        guard !scores.isEmpty, scores.count == targets.count else { return }

        // Gradient descent optimization for log loss parameters A and B
        var currA = -1.0
        var currB = 0.0

        for _ in 0..<maxIterations {
            var gradA = 0.0
            var gradB = 0.0

            for (s, t) in zip(scores, targets) {
                let logit = currA * s + currB
                let p = 1.0 / (1.0 + exp(-logit))
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

    /// Predict probability.
    /// - Parameters:
    ///   - score: The score.
    /// - Returns: A `Double` result.
    public func predictProbability(score: Double) -> Double {
        guard isFitted else { return 1.0 / (1.0 + exp(-score)) }
        let logit = a * score + b
        return 1.0 / (1.0 + exp(-logit))
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
