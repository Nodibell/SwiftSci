import Foundation

// MARK: - Kaplan-Meier Survival Estimator

/// Kaplan-Meier non-parametric statistic used to estimate the survival function from life-tables or right-censored data.
public struct KaplanMeier: Sendable {
    /// Represents a point on the Kaplan-Meier survival probability curve.
    public struct SurvivalPoint: Sendable {
        /// Time point of observation.
        public let time: Double
        /// Number of subjects at risk prior to this time point.
        public let atRisk: Int
        /// Number of observed events (e.g. deaths, failures) at this time point.
        public let events: Int
        /// Cumulative estimated survival probability S(t) at this time point.
        public let survivalProbability: Double
    }

    /// The timeline.
    public let timeline: [SurvivalPoint]

    /// Creates a new instance.
    /// - Parameters:
    ///   - durations: The durations.
    ///   - events: The events.
    public init(durations: [Double], events: [Int]) {
        precondition(durations.count == events.count, "Durations and events counts must match")

        let n = durations.count
        guard n > 0 else {
            self.timeline = []
            return
        }

        // Pair and sort by duration ascending
        let paired = zip(durations, events).sorted { $0.0 < $1.0 }

        // Group by distinct event times
        var timeMap: [Double: (events: Int, censored: Int)] = [:]
        for (d, e) in paired {
            var curr = timeMap[d, default: (events: 0, censored: 0)]
            if e == 1 {
                curr.events += 1
            } else {
                curr.censored += 1
            }
            timeMap[d] = curr
        }

        let sortedTimes = timeMap.keys.sorted()
        var currentAtRisk = n
        var currentSurvival = 1.0
        var points = [SurvivalPoint]()
        points.reserveCapacity(sortedTimes.count)

        for t in sortedTimes {
            guard let counts = timeMap[t] else { continue }
            let d = counts.events
            let c = counts.censored

            if d > 0 {
                let p = 1.0 - (Double(d) / Double(currentAtRisk))
                currentSurvival *= p
            }

            points.append(SurvivalPoint(
                time: t,
                atRisk: currentAtRisk,
                events: d,
                survivalProbability: currentSurvival
            ))

            currentAtRisk -= (d + c)
        }

        self.timeline = points
    }

    /// Evaluates estimated survival probability S(t) at given query time t.
    public func predictSurvivalProbability(at time: Double) -> Double {
        guard let firstPoint = timeline.first, time >= firstPoint.time else { return 1.0 }

        var lastProb = 1.0
        for p in timeline {
            if p.time <= time {
                lastProb = p.survivalProbability
            } else {
                break
            }
        }
        return lastProb
    }
}

// MARK: - Cox Proportional Hazards Model

/// Cox Proportional Hazards model for semi-parametric survival time regression.
public final class CoxProportionalHazards: @unchecked Sendable {
    /// The coefficients.
    public private(set) var coefficients: [Double] = []
    /// is fitted.
    public private(set) var isFitted: Bool = false

    /// Creates a new instance.
    public init() {}

    /// Fit.
    /// - Parameters:
    ///   - features: The features.
    ///   - durations: The durations.
    ///   - events: The events.
    ///   - maxIterations: The max iterations.
    ///   - lr: The lr.
    public func fit(features: [[Double]], durations: [Double], events: [Int], maxIterations: Int = 50, lr: Double = 0.01) {
        let n = features.count
        guard n > 0 && n == durations.count && n == events.count else { return }

        let p = features[0].count
        var beta = [Double](repeating: 0.0, count: p)

        // Gradient descent on Partial Log-Likelihood
        for _ in 0..<maxIterations {
            var grad = [Double](repeating: 0.0, count: p)

            for i in 0..<n where events[i] == 1 {
                let t_i = durations[i]
                let x_i = features[i]

                // Risk set R_i: samples with duration >= t_i
                var riskSum = 0.0
                var riskWeightedX = [Double](repeating: 0.0, count: p)

                for j in 0..<n where durations[j] >= t_i {
                    let dot = zip(beta, features[j]).map(*).reduce(0, +)
                    let expVal = exp(min(20.0, max(-20.0, dot)))
                    riskSum += expVal
                    for k in 0..<p {
                        riskWeightedX[k] += expVal * features[j][k]
                    }
                }

                if riskSum > 0 {
                    for k in 0..<p {
                        grad[k] += x_i[k] - (riskWeightedX[k] / riskSum)
                    }
                }
            }

            for k in 0..<p {
                beta[k] += lr * (grad[k] / Double(n))
            }
        }

        self.coefficients = beta
        self.isFitted = true
    }

    /// Predicts log hazard ratio exp(beta^T * x) for a feature vector.
    public func predictPartialHazard(features: [Double]) -> Double {
        guard isFitted, coefficients.count == features.count else { return 1.0 }
        let dot = zip(coefficients, features).map(*).reduce(0, +)
        return exp(dot)
    }
}
