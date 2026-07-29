import Foundation
import Accelerate
import SwiftStats

public actor GARCHModel {
    /// The order.
    public let order: (p: Int, q: Int) // p = GARCH order (lagged variance), q = ARCH order (lagged residuals squared)
    
    /// The omega.
    public private(set) var omega: Double = 0.0
    /// The alpha.
    public private(set) var alpha: [Double] = []
    /// The beta.
    public private(set) var beta: [Double] = []
    /// The mu.
    public private(set) var mu: Double = 0.0
    
    private var series: [Double] = []
    private var residuals: [Double] = []
    private var initialVar: Double = 1.0
    private var isFitted = false
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - p: The p.
    ///   - q: The q.
    /// - Throws: An error if the operation fails.
    public init(p: Int, q: Int) throws {
        guard p >= 0 else { throw ForecastError.invalidAROrder(p) }
        guard q >= 0 else { throw ForecastError.invalidMAOrder(q) }
        self.order = (p, q)
    }
    
    /// Fits the GARCH model on the series using Maximum Likelihood Estimation via Coordinate Descent.
    public func fit(series: [Double]) throws {
        let n = series.count
        guard n > 0 else { throw ForecastError.emptyTimeSeries }
        let minLen = order.p + order.q + 4
        guard n >= minLen else {
            throw ForecastError.insufficientLength(minimum: minLen, got: n)
        }
        
        self.series = series
        
        // Estimate mu as mean of the series
        self.mu = vDSP.mean(series)
        
        // Compute residuals
        self.residuals = series.map { $0 - mu }
        
        // Compute initial variance
        let meanRes = vDSP.mean(residuals)
        let sumSquaredDiffs = residuals.reduce(0.0) { $0 + ($1 - meanRes) * ($1 - meanRes) }
        self.initialVar = sumSquaredDiffs / Double(n - 1)
        if initialVar <= 0 {
            initialVar = 1.0
        }
        
        // Handle GARCH(0, 0)
        if order.p == 0 && order.q == 0 {
            self.omega = initialVar
            self.alpha = []
            self.beta = []
            self.isFitted = true
            return
        }
        
        // Coordinate Descent Optimization for MLE parameters
        var bestOmega = 0.1 * initialVar
        var bestAlpha = [Double](repeating: 0.1 / Double(Swift.max(order.q, 1)), count: order.q)
        var bestBeta = [Double](repeating: 0.8 / Double(Swift.max(order.p, 1)), count: order.p)
        
        var bestLikelihood = computeLogLikelihood(omega: bestOmega, alpha: bestAlpha, beta: bestBeta, eps: residuals, initialVar: initialVar)
        
        let stepSizes = [0.1, 0.01, 0.001]
        let maxIterations = 50
        
        for step in stepSizes {
            var improved = true
            var iter = 0
            while improved && iter < maxIterations {
                improved = false
                iter += 1
                
                // Try adjusting omega
                for change in [-step, step] {
                    let nextOmega = bestOmega + change * bestOmega
                    if nextOmega > 1e-5 {
                        let L = computeLogLikelihood(omega: nextOmega, alpha: bestAlpha, beta: bestBeta, eps: residuals, initialVar: initialVar)
                        if L > bestLikelihood {
                            bestLikelihood = L
                            bestOmega = nextOmega
                            improved = true
                        }
                    }
                }
                
                // Try adjusting alpha
                if order.q > 0 {
                    for i in 0..<order.q {
                        for change in [-step, step] {
                            var nextAlpha = bestAlpha
                            nextAlpha[i] += change * bestAlpha[i]
                            if nextAlpha[i] >= 0.0 && nextAlpha.reduce(0, +) + bestBeta.reduce(0, +) < 0.999 {
                                let L = computeLogLikelihood(omega: bestOmega, alpha: nextAlpha, beta: bestBeta, eps: residuals, initialVar: initialVar)
                                if L > bestLikelihood {
                                    bestLikelihood = L
                                    bestAlpha = nextAlpha
                                    improved = true
                                }
                            }
                        }
                    }
                }
                
                // Try adjusting beta
                if order.p > 0 {
                    for j in 0..<order.p {
                        for change in [-step, step] {
                            var nextBeta = bestBeta
                            nextBeta[j] += change * bestBeta[j]
                            if nextBeta[j] >= 0.0 && bestAlpha.reduce(0, +) + nextBeta.reduce(0, +) < 0.999 {
                                let L = computeLogLikelihood(omega: bestOmega, alpha: bestAlpha, beta: nextBeta, eps: residuals, initialVar: initialVar)
                                if L > bestLikelihood {
                                    bestLikelihood = L
                                    bestBeta = nextBeta
                                    improved = true
                                }
                            }
                        }
                    }
                }
            }
        }
        
        self.omega = bestOmega
        self.alpha = bestAlpha
        self.beta = bestBeta
        self.isFitted = true
    }
    
    /// Forecasts conditional volatility (standard deviation) for the specified steps.
    public func forecastVolatility(steps: Int) throws -> [Double] {
        guard isFitted else { throw ForecastError.notFitted }
        guard steps > 0 else { return [] }
        
        let n = residuals.count
        var sig2 = [Double](repeating: initialVar, count: n)
        
        // Recompute fitted variance
        for t in 0..<n {
            var v = omega
            for i in 0..<order.q {
                let idx = t - 1 - i
                let e2 = idx >= 0 ? residuals[idx] * residuals[idx] : residuals[0] * residuals[0]
                v += alpha[i] * e2
            }
            for j in 0..<order.p {
                let idx = t - 1 - j
                let s2 = idx >= 0 ? sig2[idx] : initialVar
                v += beta[j] * s2
            }
            sig2[t] = Swift.max(v, 1e-6)
        }
        
        var forecasts = [Double](repeating: 0.0, count: steps)
        var historySig2 = sig2
        var historyEps2 = residuals.map { $0 * $0 }
        
        for step in 0..<steps {
            var v = omega
            // ARCH terms
            for i in 0..<order.q {
                let idx = historyEps2.count - 1 - i
                if idx >= n {
                    let fcIdx = idx - n
                    v += alpha[i] * forecasts[fcIdx]
                } else {
                    v += alpha[i] * historyEps2[idx]
                }
            }
            // GARCH terms
            for j in 0..<order.p {
                let idx = historySig2.count - 1 - j
                if idx >= n {
                    let fcIdx = idx - n
                    v += beta[j] * forecasts[fcIdx]
                } else {
                    v += beta[j] * historySig2[idx]
                }
            }
            
            let val = Swift.max(v, 1e-6)
            forecasts[step] = val
            historySig2.append(val)
            historyEps2.append(val)
        }
        
        return forecasts.map { sqrt($0) }
    }
    
    // MARK: - Private helper methods
    
    private func computeLogLikelihood(omega: Double, alpha: [Double], beta: [Double], eps: [Double], initialVar: Double) -> Double {
        let n = eps.count
        var sig2 = [Double](repeating: initialVar, count: n)
        
        var logL = 0.0
        for t in 0..<n {
            var v = omega
            for i in 0..<order.q {
                let idx = t - 1 - i
                let e2 = idx >= 0 ? eps[idx] * eps[idx] : eps[0] * eps[0]
                v += alpha[i] * e2
            }
            for j in 0..<order.p {
                let idx = t - 1 - j
                let s2 = idx >= 0 ? sig2[idx] : initialVar
                v += beta[j] * s2
            }
            
            let variance = Swift.max(v, 1e-6)
            sig2[t] = variance
            
            let term = log(variance) + (eps[t] * eps[t]) / variance
            logL += term
        }
        
        return -0.5 * (Double(n) * log(2.0 * Double.pi) + logL)
    }
}
