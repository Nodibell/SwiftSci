import Foundation

/// Error, Trend, and Seasonal (ETS) State Space forecasting model.
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
    private var lastLevel: Double = 0.0
    private var lastTrend: Double = 0.0
    private var seasonalComponents: [Double] = []
    private var isFitted: Bool = false

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

    /// Fits the ETS model to the time series data.
    public func fit(series: [Double]) async throws {
        guard !series.isEmpty else {
            throw ForecastError.emptyTimeSeries
        }
        guard series.count >= max(3, period * 2) else {
            throw ForecastError.insufficientLength(minimum: max(3, period * 2), got: series.count)
        }

        self.fittedSeries = series
        let n = series.count

        // Initialize Level and Trend
        var l = series[0]
        var b = (series[min(n - 1, 3)] - series[0]) / 3.0
        if trendType == .none { b = 0.0 }

        // Initialize Seasonal
        var s = [Double](repeating: 1.0, count: period)
        if seasonalType == .additive {
            s = [Double](repeating: 0.0, count: period)
        } else if seasonalType == .multiplicative {
            let meanVal = series.reduce(0.0, +) / Double(n)
            s = (0..<period).map { i in meanVal == 0 ? 1.0 : series[i] / meanVal }
        }

        // Forward state update filtering
        for t in 0..<n {
            let y = series[t]
            let sIdx = t % period
            let sVal = (seasonalType != .none) ? s[sIdx] : (seasonalType == .additive ? 0.0 : 1.0)
            
            let yHat: Double
            if seasonalType == .additive {
                yHat = l + (trendType != .none ? b : 0.0) + sVal
            } else if seasonalType == .multiplicative {
                yHat = (l + (trendType != .none ? b : 0.0)) * max(1e-6, sVal)
            } else {
                yHat = l + (trendType != .none ? b : 0.0)
            }

            let e = y - yHat
            
            let prevL = l
            l = prevL + (trendType != .none ? b : 0.0) + alpha * e
            if trendType == .additive {
                b = b + beta * e
            } else if trendType == .damped {
                b = phi * b + beta * e
            }

            if seasonalType == .additive {
                s[sIdx] = sVal + gamma * e
            } else if seasonalType == .multiplicative && prevL != 0 {
                s[sIdx] = max(1e-6, sVal + gamma * (e / prevL))
            }
        }

        self.lastLevel = l
        self.lastTrend = b
        self.seasonalComponents = s
        self.isFitted = true
    }

    /// Generates future forecasts for `steps` ahead.
    public func forecast(steps: Int) async throws -> [Double] {
        guard isFitted else {
            throw ForecastError.emptyTimeSeries
        }
        guard steps > 0 else { return [] }

        var result: [Double] = []
        let curL = lastLevel
        let curB = lastTrend

        for h in 1...steps {
            let sIdx = (fittedSeries.count + h - 1) % period
            let sVal = (seasonalType != .none) ? seasonalComponents[sIdx] : (seasonalType == .additive ? 0.0 : 1.0)

            let trendFactor: Double
            if trendType == .damped {
                let phiSum = (1.0 - pow(phi, Double(h))) / (1.0 - phi)
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

    /// Automatically selects the best ETS model among combinations based on AICc.
    public static func autoFit(series: [Double], period: Int = 1) async throws -> ETSModel {
        let trends: [TrendType] = [.none, .additive, .damped]
        let seasonals: [SeasonalType] = period > 1 ? [.none, .additive, .multiplicative] : [.none]
        
        var bestModel: ETSModel?
        var minAICc = Double.infinity

        for t in trends {
            for s in seasonals {
                let model = ETSModel(error: .additive, trend: t, seasonal: s, period: period)
                do {
                    try await model.fit(series: series)
                    let n = Double(series.count)
                    let k = 3.0 // k parameters (level, trend, seasonal)
                    // Compute SSE for AICc
                    let preds = try await model.forecast(steps: 1)
                    let sse = pow(series.last! - (preds.first ?? series.last!), 2) + 1e-6
                    let aicc = n * log(sse / n) + 2.0 * k + (2.0 * k * (k + 1.0)) / max(1.0, n - k - 1.0)
                    
                    if aicc < minAICc {
                        minAICc = aicc
                        bestModel = model
                    }
                } catch {
                    continue
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
