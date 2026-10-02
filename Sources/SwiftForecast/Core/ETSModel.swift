import Foundation
import Accelerate
import SwiftStats

/// Error, Trend, and Seasonal (ETS) State Space forecasting model.
///
/// Implements point forecasting and state-space filtering for canonical ETS specifications:
/// - Error: Additive (\(A\)) or Multiplicative (\(M\))
/// - Trend: None (\(N\)), Additive (\(A\)), or Damped Additive (\(A_d\))
/// - Seasonal: None (\(N\)), Additive (\(A\)), or Multiplicative (\(M\))
public actor ETSModel {
    /// Classification for the error component (Additive or Multiplicative).
    public enum ErrorType: String, Sendable { case additive = "A", multiplicative = "M" }
    /// Classification for the trend component (None, Additive, or Damped).
    public enum TrendType: String, Sendable { case none = "N", additive = "A", damped = "Ad" }
    /// Classification for the seasonal component (None, Additive, or Multiplicative).
    public enum SeasonalType: String, Sendable { case none = "N", additive = "A", multiplicative = "M" }

    /// Configured error component specification.
    public let errorType: ErrorType
    /// Configured trend component specification.
    public let trendType: TrendType
    /// Configured seasonal component specification.
    public let seasonalType: SeasonalType
    /// Seasonal period length (number of observations per full cycle).
    public let period: Int

    private var alpha: Double = 0.3
    private var beta: Double = 0.1
    private var gamma: Double = 0.1
    private var phi: Double = 0.98

    private var fittedSeries: [Double] = []
    public private(set) var lastLevel: Double = 0.0
    public private(set) var lastTrend: Double = 0.0
    public private(set) var seasonalComponents: [Double] = []
    public private(set) var fittedValues: [Double] = []
    public private(set) var residuals: [Double] = []
    public private(set) var sse: Double = 0.0
    public private(set) var logLikelihood: Double = 0.0
    public private(set) var aic: Double = 0.0
    public private(set) var aicc: Double = 0.0
    public private(set) var bic: Double = 0.0
    public private(set) var isFitted: Bool = false

    /// Initializes an Error, Trend, and Seasonal (ETS) forecasting model.
    /// - Parameters:
    ///   - error: Error model component (`.additive` or `.multiplicative`). Defaults to `.additive`.
    ///   - trend: Trend component variant (`.none`, `.additive`, or `.damped`). Defaults to `.additive`.
    ///   - seasonal: Seasonal cycle variant (`.none`, `.additive`, or `.multiplicative`). Defaults to `.none`.
    ///   - period: Number of periods per seasonal cycle. Defaults to 1.
    public init(
        error: ErrorType = .additive,
        trend: TrendType = .additive,
        seasonal: SeasonalType = .none,
        period: Int = 1
    ) {
        self.errorType = error
        self.trendType = trend
        self.seasonalType = seasonal
        self.period = max(1, period)
    }

    /// Fits the ETS model to the time series data using recursive state-space filtering.
    /// - Parameters:
    ///   - series: 1D temporal time-series observations array.
    /// - Throws: `ForecastError` if series length is insufficient, values contain NaNs/Infs, or positive domain violated.
    public func fit(series: [Double]) async throws {
        guard !series.isEmpty else {
            throw ForecastError.emptyTimeSeries
        }
        if series.contains(where: { $0.isNaN }) {
            throw ForecastError.containsNaN
        }
        if series.contains(where: { $0.isInfinite }) {
            throw ForecastError.containsInfinity
        }
        guard series.count >= max(3, period * 2) else {
            throw ForecastError.insufficientLength(minimum: max(3, period * 2), got: series.count)
        }

        if errorType == .multiplicative || seasonalType == .multiplicative {
            guard series.allSatisfy({ $0 > 0.0 }) else {
                throw ForecastError.invalidParameter("Multiplicative ETS components require strictly positive values.")
            }
        }

        self.fittedSeries = series
        let n = series.count

        // Initialize Level and Trend
        var l = series[0]
        var b = (series[min(n - 1, 3)] - series[0]) / 3.0
        if trendType == .none { b = 0.0 }

        // Initialize Seasonal Components
        var s = [Double](repeating: 1.0, count: period)
        if seasonalType == .additive {
            s = [Double](repeating: 0.0, count: period)
        } else if seasonalType == .multiplicative {
            let meanVal = try Stats.mean(series)
            s = (0..<period).map { i in meanVal == 0 ? 1.0 : series[i] / meanVal }
        }

        var inSampleFitted = [Double](repeating: 0.0, count: n)
        var inSampleResiduals = [Double](repeating: 0.0, count: n)
        var totalSSE = 0.0
        var logMuSum = 0.0

        // Forward state update filtering
        for t in 0..<n {
            let y = series[t]
            let sIdx = t % period
            let sVal = (seasonalType != .none) ? s[sIdx] : (seasonalType == .additive ? 0.0 : 1.0)

            let prevL = l
            let prevB = b

            // Trend component preceding observation t
            let trendComp: Double
            switch trendType {
            case .none: trendComp = 0.0
            case .additive: trendComp = prevB
            case .damped: trendComp = phi * prevB
            }

            // Expected point prediction mu_t
            let mu: Double
            switch seasonalType {
            case .none:
                mu = prevL + trendComp
            case .additive:
                mu = prevL + trendComp + sVal
            case .multiplicative:
                mu = (prevL + trendComp) * max(1e-6, sVal)
            }

            inSampleFitted[t] = mu

            // Error computation and state updates according to errorType
            switch errorType {
            case .additive:
                let e = y - mu
                inSampleResiduals[t] = e
                totalSSE += e * e

                // Level update: l_t = l_{t-1} + trendComp + alpha * e_t
                l = prevL + trendComp + alpha * e

                // Trend update
                switch trendType {
                case .none:
                    b = 0.0
                case .additive:
                    b = prevB + beta * e
                case .damped:
                    b = phi * prevB + beta * e
                }

                // Seasonal update
                switch seasonalType {
                case .none:
                    break
                case .additive:
                    s[sIdx] = sVal + gamma * e
                case .multiplicative:
                    let denom = max(1e-6, abs(prevL + trendComp))
                    s[sIdx] = max(1e-6, sVal + gamma * (e / denom))
                }

            case .multiplicative:
                let safeMu = max(1e-8, mu)
                let eps = (y - safeMu) / safeMu
                inSampleResiduals[t] = y - safeMu
                totalSSE += eps * eps
                logMuSum += log(safeMu)

                // Level update: l_t = (l_{t-1} + trendComp) * (1 + alpha * eps_t)
                l = (prevL + trendComp) * (1.0 + alpha * eps)

                // Trend update
                switch trendType {
                case .none:
                    b = 0.0
                case .additive:
                    b = prevB + beta * (prevL + trendComp) * eps
                case .damped:
                    b = phi * prevB + beta * (prevL + trendComp) * eps
                }

                // Seasonal update
                switch seasonalType {
                case .none:
                    break
                case .additive:
                    s[sIdx] = sVal + gamma * safeMu * eps
                case .multiplicative:
                    s[sIdx] = max(1e-6, sVal * (1.0 + gamma * eps))
                }
            }
        }

        self.lastLevel = l
        self.lastTrend = b
        self.seasonalComponents = s
        self.fittedValues = inSampleFitted
        self.residuals = inSampleResiduals
        self.sse = totalSSE

        // Free parameters in state space estimation (smoothing parameters + residual variance)
        var k = 1 // alpha
        if trendType != .none {
            k += (trendType == .damped ? 2 : 1) // beta (+ phi if damped)
        }
        if seasonalType != .none {
            k += 1 // gamma
        }
        k += 1 // sigma^2

        let nDouble = Double(n)
        let kDouble = Double(k)
        let sigma2 = max(1e-12, totalSSE / nDouble)

        let ll: Double
        if errorType == .additive {
            ll = -0.5 * nDouble * (log(2.0 * Double.pi) + log(sigma2) + 1.0)
        } else {
            ll = -0.5 * nDouble * (log(2.0 * Double.pi) + log(sigma2) + 1.0) - logMuSum
        }

        self.logLikelihood = ll
        self.aic = 2.0 * kDouble - 2.0 * ll
        let denomAICc = max(1.0, nDouble - kDouble - 1.0)
        self.aicc = self.aic + (2.0 * kDouble * (kDouble + 1.0)) / denomAICc
        self.bic = kDouble * log(nDouble) - 2.0 * ll
        self.isFitted = true
    }

    /// Generates future forecasts for `steps` ahead.
    /// - Parameters:
    ///   - steps: Number of future time steps to forecast. Must be at least 1.
    /// - Throws: `ForecastError.notFitted` if model is unfitted, or `ForecastError.invalidHorizon` if steps < 1.
    /// - Returns: Array of forecasted future values in original series scale.
    public func forecast(steps: Int) async throws -> [Double] {
        guard isFitted else {
            throw ForecastError.notFitted
        }
        guard steps >= 1 else {
            throw ForecastError.invalidHorizon(steps)
        }

        var result: [Double] = []
        let curL = lastLevel
        let curB = lastTrend

        for h in 1...steps {
            let sIdx = (fittedSeries.count + h - 1) % period
            let sVal = (seasonalType != .none) ? seasonalComponents[sIdx] : (seasonalType == .additive ? 0.0 : 1.0)

            let trendFactor: Double
            if trendType == .damped {
                let phiSum = phi * (1.0 - pow(phi, Double(h))) / (1.0 - phi)
                trendFactor = phiSum * curB
            } else if trendType == .additive {
                trendFactor = Double(h) * curB
            } else {
                trendFactor = 0.0
            }

            let yPred: Double
            if seasonalType == .additive {
                yPred = curL + trendFactor + sVal
            } else if seasonalType == .multiplicative {
                yPred = (curL + trendFactor) * sVal
            } else {
                yPred = curL + trendFactor
            }

            result.append(yPred)
        }

        return result
    }

    /// Automatically selects the best ETS model configuration based on sample-wide AICc minimization.
    /// - Parameters:
    ///   - series: 1D temporal time-series observations array.
    ///   - period: Seasonal cycle periodicity (number of observations per season).
    /// - Throws: `ForecastError` if series length is insufficient or values contain NaNs.
    /// - Returns: Fitted Exponential Smoothing state-space model minimizing AICc.
    public static func autoFit(series: [Double], period: Int = 1) async throws -> ETSModel {
        let trends: [TrendType] = [.none, .additive, .damped]
        let seasonals: [SeasonalType] = period > 1 ? [.none, .additive, .multiplicative] : [.none]
        let isStrictlyPositive = series.allSatisfy { $0 > 0.0 }
        let errors: [ErrorType] = isStrictlyPositive ? [.additive, .multiplicative] : [.additive]

        var bestModel: ETSModel?
        var minAICc = Double.infinity

        for e in errors {
            for t in trends {
                for s in seasonals {
                    // Skip multiplicative seasonal on non-positive series
                    if s == .multiplicative && !isStrictlyPositive { continue }

                    let model = ETSModel(error: e, trend: t, seasonal: s, period: period)
                    do {
                        try await model.fit(series: series)
                        let candidateAICc = await model.aicc
                        if candidateAICc.isFinite && !candidateAICc.isNaN && candidateAICc < minAICc {
                            minAICc = candidateAICc
                            bestModel = model
                        }
                    } catch {
                        continue
                    }
                }
            }
        }

        if let best = bestModel {
            return best
        }

        let fallback = ETSModel(error: .additive, trend: .additive, seasonal: .none, period: period)
        try await fallback.fit(series: series)
        return fallback
    }
}
