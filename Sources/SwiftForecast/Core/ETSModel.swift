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

    /// Estimated level smoothing parameter alpha in (0, 1).
    public private(set) var alpha: Double
    /// Estimated trend smoothing parameter beta in (0, 1).
    public private(set) var beta: Double
    /// Estimated seasonal smoothing parameter gamma in (0, 1).
    public private(set) var gamma: Double
    /// Estimated damping parameter phi in (0, 1).
    public private(set) var phi: Double

    private var fittedSeries: [Double] = []
    public private(set) var lastLevel: Double = 0.0
    public private(set) var lastTrend: Double = 0.0
    public private(set) var seasonalComponents: [Double] = []
    public private(set) var fittedValues: [Double] = []
    /// In-sample forecast residuals: e_t = y_t - mu_t.
    public private(set) var residuals: [Double] = []
    /// Normalized state-space innovations: eps_t = (y_t - mu_t) / mu_t for multiplicative, e_t for additive.
    public private(set) var innovations: [Double] = []
    /// Sum of squared innovations sum(eps_t^2) utilized in state-space Gaussian likelihood.
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
    ///   - alpha: Initial smoothing parameter for level. Defaults to 0.3.
    ///   - beta: Initial smoothing parameter for trend. Defaults to 0.1.
    ///   - gamma: Initial smoothing parameter for seasonality. Defaults to 0.1.
    ///   - phi: Initial damping coefficient. Defaults to 0.98.
    public init(
        error: ErrorType = .additive,
        trend: TrendType = .additive,
        seasonal: SeasonalType = .none,
        period: Int = 1,
        alpha: Double = 0.3,
        beta: Double = 0.1,
        gamma: Double = 0.1,
        phi: Double = 0.98
    ) {
        self.errorType = error
        self.trendType = trend
        self.seasonalType = seasonal
        self.period = max(1, period)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi
    }

    /// Fits the ETS model to the time series data using recursive state-space filtering.
    /// - Parameters:
    ///   - series: 1D temporal time-series observations array.
    ///   - optimizeParameters: When true, optimizes smoothing parameters via Nelder-Mead simplex search.
    /// - Throws: `ForecastError` if series length is insufficient, values contain NaNs/Infs, or positive domain violated.
    public func fit(series: [Double], optimizeParameters: Bool = true) async throws {
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

        // Optimize smoothing parameters if enabled
        if optimizeParameters {
            var initialGuess: [Double] = [self.alpha]
            var lowerBounds: [Double] = [0.001]
            var upperBounds: [Double] = [0.99]

            if trendType != .none {
                initialGuess.append(self.beta)
                lowerBounds.append(0.0001)
                upperBounds.append(0.99)
            }
            if seasonalType != .none {
                initialGuess.append(self.gamma)
                lowerBounds.append(0.0001)
                upperBounds.append(0.99)
            }
            if trendType == .damped {
                initialGuess.append(self.phi)
                lowerBounds.append(0.8)
                upperBounds.append(0.99)
            }

            let opt = NelderMead(maxIterations: 80, tolerance: 1e-4)
            let bestParams = opt.minimize(
                objective: { params in
                    var idx = 0
                    let a = params[idx]
                    idx += 1
                    let b = self.trendType != .none ? params[idx] : self.beta
                    if self.trendType != .none { idx += 1 }
                    let g = self.seasonalType != .none ? params[idx] : self.gamma
                    if self.seasonalType != .none { idx += 1 }
                    let p = self.trendType == .damped ? params[idx] : self.phi

                    guard let filtered = try? self.filterTrajectory(
                        series: series,
                        alphaVal: a,
                        betaVal: b,
                        gammaVal: g,
                        phiVal: p
                    ) else {
                        return Double.infinity
                    }
                    return filtered.totalSSE
                },
                initialGuess: initialGuess,
                lowerBounds: lowerBounds,
                upperBounds: upperBounds
            )

            var idx = 0
            self.alpha = bestParams[idx]
            idx += 1
            if self.trendType != .none {
                self.beta = bestParams[idx]
                idx += 1
            }
            if self.seasonalType != .none {
                self.gamma = bestParams[idx]
                idx += 1
            }
            if self.trendType == .damped {
                self.phi = bestParams[idx]
            }
        }

        let filtered = try filterTrajectory(
            series: series,
            alphaVal: self.alpha,
            betaVal: self.beta,
            gammaVal: self.gamma,
            phiVal: self.phi
        )

        self.lastLevel = filtered.lastL
        self.lastTrend = filtered.lastB
        self.seasonalComponents = filtered.seasonal
        self.fittedValues = filtered.fitted
        self.residuals = filtered.residuals
        self.innovations = filtered.innovations
        self.sse = filtered.totalSSE

        // Free parameters in state space estimation (smoothing parameters + residual variance)
        var k = 1 // alpha
        if trendType != .none {
            k += (trendType == .damped ? 2 : 1) // beta (+ phi if damped)
        }
        if seasonalType != .none {
            k += 1 // gamma
        }
        k += 1 // sigma^2

        let n = series.count
        let nDouble = Double(n)
        let kDouble = Double(k)
        let sigma2 = max(1e-12, filtered.totalSSE / nDouble)

        let ll: Double
        if errorType == .additive {
            ll = -0.5 * nDouble * (log(2.0 * Double.pi) + log(sigma2) + 1.0)
        } else {
            ll = -0.5 * nDouble * (log(2.0 * Double.pi) + log(sigma2) + 1.0) - filtered.logMuSum
        }

        self.logLikelihood = ll
        self.aic = 2.0 * kDouble - 2.0 * ll
        let denomAICc = max(1.0, nDouble - kDouble - 1.0)
        self.aicc = self.aic + (2.0 * kDouble * (kDouble + 1.0)) / denomAICc
        self.bic = kDouble * log(nDouble) - 2.0 * ll
        self.isFitted = true
    }

    private func filterTrajectory(
        series: [Double],
        alphaVal: Double,
        betaVal: Double,
        gammaVal: Double,
        phiVal: Double
    ) throws -> (
        totalSSE: Double,
        logMuSum: Double,
        lastL: Double,
        lastB: Double,
        seasonal: [Double],
        fitted: [Double],
        residuals: [Double],
        innovations: [Double]
    ) {
        let n = series.count

        // Initialize Level and Trend
        var l = series[0]
        var b = 0.0
        if trendType != .none {
            if period > 1 && n >= 2 * period {
                let mean1 = series[0..<period].reduce(0.0, +) / Double(period)
                let mean2 = series[period..<(2 * period)].reduce(0.0, +) / Double(period)
                b = (mean2 - mean1) / Double(period)
            } else {
                let denom = Double(max(1, min(n - 1, 3)))
                b = (series[min(n - 1, 3)] - series[0]) / denom
            }
        }

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
        var inSampleInnovations = [Double](repeating: 0.0, count: n)
        var totalSSE = 0.0
        var logMuSum = 0.0

        for t in 0..<n {
            let y = series[t]
            let sIdx = t % period
            let sVal = (seasonalType != .none) ? s[sIdx] : (seasonalType == .additive ? 0.0 : 1.0)

            let prevL = l
            let prevB = b

            let trendComp: Double
            switch trendType {
            case .none: trendComp = 0.0
            case .additive: trendComp = prevB
            case .damped: trendComp = phiVal * prevB
            }

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
            inSampleResiduals[t] = y - mu

            switch errorType {
            case .additive:
                let e = y - mu
                inSampleInnovations[t] = e
                totalSSE += e * e

                l = prevL + trendComp + alphaVal * e

                switch trendType {
                case .none:
                    b = 0.0
                case .additive:
                    b = prevB + betaVal * e
                case .damped:
                    b = phiVal * prevB + betaVal * e
                }

                switch seasonalType {
                case .none:
                    break
                case .additive:
                    s[sIdx] = sVal + gammaVal * e
                case .multiplicative:
                    let denom = max(1e-6, abs(prevL + trendComp))
                    s[sIdx] = max(1e-6, sVal + gammaVal * (e / denom))
                }

            case .multiplicative:
                let safeMu = max(1e-8, mu)
                let eps = (y - safeMu) / safeMu
                inSampleInnovations[t] = eps
                totalSSE += eps * eps
                logMuSum += log(safeMu)

                l = (prevL + trendComp) * (1.0 + alphaVal * eps)

                switch trendType {
                case .none:
                    b = 0.0
                case .additive:
                    b = prevB + betaVal * (prevL + trendComp) * eps
                case .damped:
                    b = phiVal * prevB + betaVal * (prevL + trendComp) * eps
                }

                switch seasonalType {
                case .none:
                    break
                case .additive:
                    s[sIdx] = sVal + gammaVal * safeMu * eps
                case .multiplicative:
                    s[sIdx] = max(1e-6, sVal * (1.0 + gammaVal * eps))
                }
            }
        }

        return (
            totalSSE: totalSSE,
            logMuSum: logMuSum,
            lastL: l,
            lastB: b,
            seasonal: s,
            fitted: inSampleFitted,
            residuals: inSampleResiduals,
            innovations: inSampleInnovations
        )
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
