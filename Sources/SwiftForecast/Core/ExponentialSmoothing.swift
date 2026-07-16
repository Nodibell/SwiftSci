import Foundation
import Accelerate

public enum SmoothingMethod: Sendable {
    case simple
    case double(beta: Double)
    case holtWinters(beta: Double, gamma: Double, period: Int, seasonal: DecompositionModel)
}

public actor ExponentialSmoothing {
    public let method: SmoothingMethod
    private var alphaInput: Double?
    
    private var alpha: Double = 0.5
    private var beta: Double = 0.1
    private var gamma: Double = 0.1
    
    private var series: [Double] = []
    private var level: [Double] = []
    private var trend: [Double] = []
    private var seasonal: [Double] = []
    private var fitted: [Double] = []
    private var isFitted = false
    
    public init(method: SmoothingMethod, alpha: Double? = nil) {
        self.method = method
        self.alphaInput = alpha
    }
    
    /// Fits the exponential smoothing model to the series.
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
        
        let minLen: Int
        switch method {
        case .simple: minLen = 2
        case .double: minLen = 3
        case let .holtWinters(_, _, period, _): minLen = period * 2
        }
        
        guard series.count >= minLen else {
            throw ForecastError.insufficientLength(minimum: minLen, got: series.count)
        }
        
        self.series = series
        
        // Optimize hyperparameters if needed
        try await optimizeParameters()
        
        // Final run with optimized parameters
        try runSmoothing(alpha: self.alpha, beta: self.beta, gamma: self.gamma)
        self.isFitted = true
    }
    
    /// Forecast horizon steps ahead.
    public func forecast(horizon: Int) throws -> ForecastResult {
        guard isFitted else {
            throw ForecastError.notFitted
        }
        guard horizon >= 1 else {
            throw ForecastError.invalidHorizon(horizon)
        }
        
        let n = series.count
        var preds = [Double](repeating: 0.0, count: horizon)
        
        let lastLevel = level.last ?? series[0]
        let lastTrend: Double
        if case .simple = method {
            lastTrend = 0.0
        } else {
            lastTrend = trend.last ?? 0.0
        }
        
        switch method {
        case .simple:
            for h in 0..<horizon {
                preds[h] = lastLevel
            }
        case .double:
            for h in 0..<horizon {
                preds[h] = lastLevel + Double(h + 1) * lastTrend
            }
        case let .holtWinters(_, _, period, model):
            for h in 0..<horizon {
                let step = h + 1
                let base = lastLevel + Double(step) * lastTrend
                let seasonalIdx = (n - period + h) % period
                let seasonalFactor = seasonal[n - period + seasonalIdx]
                if model == .additive {
                    preds[h] = base + seasonalFactor
                } else {
                    preds[h] = base * seasonalFactor
                }
            }
        }
        
        // Compute metrics on training set
        var residuals = [Double](repeating: 0.0, count: n)
        vDSP.subtract(series, fitted, result: &residuals)
        
        let mse = vDSP.sumOfSquares(residuals) / Double(n)
        let mae = residuals.reduce(0.0) { $0 + abs($1) } / Double(n)
        
        return ForecastResult(
            predictions: preds,
            lowerBound: nil,
            upperBound: nil,
            fittedValues: fitted,
            residuals: residuals,
            aic: nil,
            mse: mse,
            mae: mae
        )
    }
    
    public func fittedValues() throws -> [Double] {
        guard isFitted else {
            throw ForecastError.notFitted
        }
        return fitted
    }
    
    // MARK: - Private helper methods
    
    private func optimizeParameters() async throws {
        // Simple ES optimization
        if case .simple = method {
            if let fixedAlpha = alphaInput {
                guard fixedAlpha > 0 && fixedAlpha < 1 else {
                    throw ForecastError.invalidAlpha(fixedAlpha)
                }
                self.alpha = fixedAlpha
                return
            }
            
            // Search alpha over grid [0.1, 0.2, ..., 0.9]
            var bestAlpha = 0.5
            var minMse = Double.greatestFiniteMagnitude
            
            for a in stride(from: 0.1, through: 0.9, by: 0.1) {
                do {
                    try runSmoothing(alpha: a, beta: 0.0, gamma: 0.0)
                    let mse = try computeTrainingMse()
                    if mse < minMse {
                        minMse = mse
                        bestAlpha = a
                    }
                } catch {}
            }
            self.alpha = bestAlpha
        }
        
        // Double ES optimization
        if case let .double(fixedBeta) = method {
            guard fixedBeta > 0 && fixedBeta < 1 else {
                throw ForecastError.invalidBeta(fixedBeta)
            }
            self.beta = fixedBeta
            
            if let fixedAlpha = alphaInput {
                guard fixedAlpha > 0 && fixedAlpha < 1 else {
                    throw ForecastError.invalidAlpha(fixedAlpha)
                }
                self.alpha = fixedAlpha
                return
            }
            
            // Search alpha
            var bestAlpha = 0.5
            var minMse = Double.greatestFiniteMagnitude
            for a in stride(from: 0.1, through: 0.9, by: 0.1) {
                do {
                    try runSmoothing(alpha: a, beta: fixedBeta, gamma: 0.0)
                    let mse = try computeTrainingMse()
                    if mse < minMse {
                        minMse = mse
                        bestAlpha = a
                    }
                } catch {}
            }
            self.alpha = bestAlpha
        }
        
        // Holt-Winters optimization
        if case let .holtWinters(fixedBeta, fixedGamma, _, _) = method {
            guard fixedBeta > 0 && fixedBeta < 1 else {
                throw ForecastError.invalidBeta(fixedBeta)
            }
            guard fixedGamma > 0 && fixedGamma < 1 else {
                throw ForecastError.invalidGamma(fixedGamma)
            }
            self.beta = fixedBeta
            self.gamma = fixedGamma
            
            if let fixedAlpha = alphaInput {
                guard fixedAlpha > 0 && fixedAlpha < 1 else {
                    throw ForecastError.invalidAlpha(fixedAlpha)
                }
                self.alpha = fixedAlpha
                return
            }
            
            // Grid search alpha
            var bestAlpha = 0.5
            var minMse = Double.greatestFiniteMagnitude
            for a in stride(from: 0.1, through: 0.9, by: 0.2) {
                do {
                    try runSmoothing(alpha: a, beta: fixedBeta, gamma: fixedGamma)
                    let mse = try computeTrainingMse()
                    if mse < minMse {
                        minMse = mse
                        bestAlpha = a
                    }
                } catch {}
            }
            self.alpha = bestAlpha
        }
    }
    
    private func computeTrainingMse() throws -> Double {
        let n = series.count
        var res = [Double](repeating: 0.0, count: n)
        vDSP.subtract(series, fitted, result: &res)
        return vDSP.sumOfSquares(res) / Double(n)
    }
    
    private func runSmoothing(alpha: Double, beta: Double, gamma: Double) throws {
        let n = series.count
        level = [Double](repeating: 0.0, count: n)
        trend = [Double](repeating: 0.0, count: n)
        fitted = [Double](repeating: 0.0, count: n)
        
        switch method {
        case .simple:
            // SES initialization
            level[0] = series[0]
            fitted[0] = series[0]
            for t in 1..<n {
                level[t] = alpha * series[t] + (1.0 - alpha) * level[t-1]
                fitted[t] = level[t-1]
            }
            
        case .double:
            // Holt's initialization
            level[0] = series[0]
            trend[0] = series[1] - series[0]
            fitted[0] = series[0]
            fitted[1] = series[0] + trend[0]
            
            for t in 1..<n {
                level[t] = alpha * series[t] + (1.0 - alpha) * (level[t-1] + trend[t-1])
                trend[t] = beta * (level[t] - level[t-1]) + (1.0 - beta) * trend[t-1]
                if t < n - 1 {
                    fitted[t+1] = level[t] + trend[t]
                }
            }
            
        case let .holtWinters(_, _, period, model):
            // Holt-Winters initialization
            seasonal = [Double](repeating: 0.0, count: n)
            
            // Initial level
            let level0 = vDSP.mean(series[0..<period])
            
            // Initial trend
            let level1 = vDSP.mean(series[period..<(period * 2)])
            let trend0 = (level1 - level0) / Double(period)
            
            level[period - 1] = level0
            trend[period - 1] = trend0
            
            // Initial seasonal components for first period
            for i in 0..<period {
                if model == .additive {
                    seasonal[i] = series[i] - level0
                } else {
                    seasonal[i] = level0 > 0 ? series[i] / level0 : 1.0
                }
            }
            
            // Fitted values for initial period
            for i in 0..<period {
                fitted[i] = series[i]
            }
            
            for t in period..<n {
                let obs = series[t]
                let prevLevel = level[t-1]
                let prevTrend = trend[t-1]
                let prevSeasonal = seasonal[t - period]
                
                let curLevel: Double
                if model == .additive {
                    curLevel = alpha * (obs - prevSeasonal) + (1.0 - alpha) * (prevLevel + prevTrend)
                } else {
                    let adjustedObs = abs(prevSeasonal) > 1e-15 ? obs / prevSeasonal : obs
                    curLevel = alpha * adjustedObs + (1.0 - alpha) * (prevLevel + prevTrend)
                }
                level[t] = curLevel
                
                let curTrend = beta * (curLevel - prevLevel) + (1.0 - beta) * prevTrend
                trend[t] = curTrend
                
                let curSeasonal: Double
                if model == .additive {
                    curSeasonal = gamma * (obs - curLevel) + (1.0 - gamma) * prevSeasonal
                } else {
                    let ratio = abs(curLevel) > 1e-15 ? obs / curLevel : 1.0
                    curSeasonal = gamma * ratio + (1.0 - gamma) * prevSeasonal
                }
                seasonal[t] = curSeasonal
                
                // Fitted value prediction for t
                let base = prevLevel + prevTrend
                if model == .additive {
                    fitted[t] = base + prevSeasonal
                } else {
                    fitted[t] = base * prevSeasonal
                }
            }
        }
    }
}
