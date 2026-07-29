import Foundation
import Accelerate
import SwiftStats

public actor SARIMAModel {
    /// The order.
    public let order: (p: Int, d: Int, q: Int)
    /// The seasonal order.
    public let seasonalOrder: (P: Int, D: Int, Q: Int, s: Int)
    
    private var arCoefficients: [Double] = []
    private var maCoefficients: [Double] = []
    private var seasonalArCoefficients: [Double] = []
    private var seasonalMaCoefficients: [Double] = []
    private var intercept: Double = 0.0
    
    private var series: [Double] = []
    private var diffSeries: [Double] = []
    private var fittedValues: [Double] = []
    private var residuals: [Double] = []
    private var isFitted = false
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - p: The p.
    ///   - d: The d.
    ///   - q: The q.
    ///   - P: The p.
    ///   - D: The d.
    ///   - Q: The q.
    ///   - s: The s.
    /// - Throws: An error if the operation fails.
    public init(p: Int, d: Int, q: Int, P: Int, D: Int, Q: Int, s: Int) throws {
        guard p >= 0 else { throw ForecastError.invalidAROrder(p) }
        guard d >= 0 else { throw ForecastError.invalidDifferencing(d) }
        guard q >= 0 else { throw ForecastError.invalidMAOrder(q) }
        guard P >= 0 else { throw ForecastError.invalidAROrder(P) }
        guard D >= 0 else { throw ForecastError.invalidDifferencing(D) }
        guard Q >= 0 else { throw ForecastError.invalidMAOrder(Q) }
        guard s >= 1 else { throw ForecastError.invalidSeasonalPeriod(s) }
        
        self.order = (p, d, q)
        self.seasonalOrder = (P, D, Q, s)
    }
    
    /// Fit.
    /// - Parameters:
    ///   - series: The series.
    /// - Throws: An error if the operation fails.
    public func fit(series: [Double]) throws {
        let n = series.count
        guard n > 0 else { throw ForecastError.emptyTimeSeries }
        
        let minLen = order.p + order.d + order.q + (seasonalOrder.P + seasonalOrder.D + seasonalOrder.Q) * seasonalOrder.s + 2
        guard n >= minLen else {
            throw ForecastError.insufficientLength(minimum: minLen, got: n)
        }
        
        self.series = series
        
        // 1. Difference the series
        var current = series
        // Seasonal differencing D times
        for _ in 0..<seasonalOrder.D {
            guard current.count > seasonalOrder.s else { break }
            var diff = [Double](repeating: 0.0, count: current.count - seasonalOrder.s)
            for i in 0..<(current.count - seasonalOrder.s) {
                diff[i] = current[i + seasonalOrder.s] - current[i]
            }
            current = diff
        }
        // Non-seasonal differencing d times
        for _ in 0..<order.d {
            guard current.count > 1 else { break }
            var diff = [Double](repeating: 0.0, count: current.count - 1)
            for i in 0..<(current.count - 1) {
                diff[i] = current[i + 1] - current[i]
            }
            current = diff
        }
        self.diffSeries = current
        
        let nDiff = diffSeries.count
        let s = seasonalOrder.s
        
        // Fit a high-order AR model to estimate initial residuals
        let k = Swift.max(order.p + order.q + (seasonalOrder.P + seasonalOrder.Q) * s, 4)
        guard nDiff > k + 2 else {
            throw ForecastError.insufficientLength(minimum: k + order.d + seasonalOrder.D * s + 3, got: n)
        }
        
        let highArCoeffs = try fitAR(series: diffSeries, order: k)
        var estResiduals = [Double](repeating: 0.0, count: nDiff)
        for t in k..<nDiff {
            var val = highArCoeffs[0] // intercept
            for i in 0..<k {
                val += highArCoeffs[1 + i] * diffSeries[t - 1 - i]
            }
            estResiduals[t] = diffSeries[t] - val
        }
        
        // Fit SARIMA using Least Squares
        let startIdx = Swift.max(
            Swift.max(order.p, order.q),
            Swift.max(seasonalOrder.P * s, seasonalOrder.Q * s)
        )
        let numSamples = nDiff - startIdx
        let numCoeffs = 1 + order.p + order.q + seasonalOrder.P + seasonalOrder.Q
        
        var yVec = [Double](repeating: 0.0, count: numSamples)
        var XMat = [Double](repeating: 0.0, count: numSamples * numCoeffs)
        
        for i in 0..<numSamples {
            let t = i + startIdx
            yVec[i] = diffSeries[t]
            
            XMat[i * numCoeffs + 0] = 1.0 // intercept
            
            var colIdx = 1
            // Non-seasonal AR
            for j in 0..<order.p {
                XMat[i * numCoeffs + colIdx] = diffSeries[t - 1 - j]
                colIdx += 1
            }
            // Seasonal AR
            for j in 0..<seasonalOrder.P {
                XMat[i * numCoeffs + colIdx] = diffSeries[t - (j + 1) * s]
                colIdx += 1
            }
            // Non-seasonal MA
            for j in 0..<order.q {
                XMat[i * numCoeffs + colIdx] = estResiduals[t - 1 - j]
                colIdx += 1
            }
            // Seasonal MA
            for j in 0..<seasonalOrder.Q {
                XMat[i * numCoeffs + colIdx] = estResiduals[t - (j + 1) * s]
                colIdx += 1
            }
        }
        
        let coeffs = try solveLeastSquares(X: XMat, y: yVec, rows: numSamples, cols: numCoeffs)
        
        self.intercept = coeffs[0]
        var coeffIdx = 1
        
        if order.p > 0 {
            self.arCoefficients = Array(coeffs[coeffIdx..<(coeffIdx + order.p)])
            coeffIdx += order.p
        }
        if seasonalOrder.P > 0 {
            self.seasonalArCoefficients = Array(coeffs[coeffIdx..<(coeffIdx + seasonalOrder.P)])
            coeffIdx += seasonalOrder.P
        }
        if order.q > 0 {
            self.maCoefficients = Array(coeffs[coeffIdx..<(coeffIdx + order.q)])
            coeffIdx += order.q
        }
        if seasonalOrder.Q > 0 {
            self.seasonalMaCoefficients = Array(coeffs[coeffIdx..<(coeffIdx + seasonalOrder.Q)])
            coeffIdx += seasonalOrder.Q
        }
        
        // Reconstruct fitted values
        var diffFitted = [Double](repeating: 0.0, count: nDiff)
        var diffResiduals = [Double](repeating: 0.0, count: nDiff)
        
        for t in 0..<nDiff {
            if t < startIdx {
                diffFitted[t] = diffSeries[t]
                diffResiduals[t] = 0.0
            } else {
                var val = intercept
                for j in 0..<order.p {
                    val += arCoefficients[j] * diffSeries[t - 1 - j]
                }
                for j in 0..<seasonalOrder.P {
                    val += seasonalArCoefficients[j] * diffSeries[t - (j + 1) * s]
                }
                for j in 0..<order.q {
                    val += maCoefficients[j] * diffResiduals[t - 1 - j]
                }
                for j in 0..<seasonalOrder.Q {
                    val += seasonalMaCoefficients[j] * diffResiduals[t - (j + 1) * s]
                }
                diffFitted[t] = val
                diffResiduals[t] = diffSeries[t] - val
            }
        }
        
        self.residuals = diffResiduals
        self.fittedValues = try integrate(fittedDiff: diffFitted, original: series, d: order.d, D: seasonalOrder.D, s: s)
        self.isFitted = true
    }
    
    /// Forecast.
    /// - Parameters:
    ///   - steps: The steps.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[Double]` result.
    public func forecast(steps: Int) throws -> [Double] {
        guard isFitted else { throw ForecastError.notFitted }
        guard steps > 0 else { return [] }
        
        let s = seasonalOrder.s
        var fcDiff = [Double](repeating: 0.0, count: steps)
        var historyDiff = diffSeries
        var historyResiduals = residuals
        
        for step in 0..<steps {
            var val = intercept
            
            // Non-seasonal AR
            for j in 0..<order.p {
                let idx = historyDiff.count - 1 - j
                val += arCoefficients[j] * (idx >= 0 ? historyDiff[idx] : 0.0)
            }
            // Seasonal AR
            for j in 0..<seasonalOrder.P {
                let idx = historyDiff.count - (j + 1) * s
                val += seasonalArCoefficients[j] * (idx >= 0 ? historyDiff[idx] : 0.0)
            }
            // Non-seasonal MA
            for j in 0..<order.q {
                let idx = historyResiduals.count - 1 - j
                val += maCoefficients[j] * (idx >= 0 ? historyResiduals[idx] : 0.0)
            }
            // Seasonal MA
            for j in 0..<seasonalOrder.Q {
                let idx = historyResiduals.count - (j + 1) * s
                val += seasonalMaCoefficients[j] * (idx >= 0 ? historyResiduals[idx] : 0.0)
            }
            
            fcDiff[step] = val
            historyDiff.append(val)
            historyResiduals.append(0.0)
        }
        
        return try integrate(fittedDiff: fcDiff, original: series, d: order.d, D: seasonalOrder.D, s: s, isForecast: true)
    }
    
    // MARK: - Private helper methods
    
    private func fitAR(series: [Double], order k: Int) throws -> [Double] {
        let n = series.count
        let numSamples = n - k
        let numCoeffs = 1 + k
        
        var yVec = [Double](repeating: 0.0, count: numSamples)
        var XMat = [Double](repeating: 0.0, count: numSamples * numCoeffs)
        
        for i in 0..<numSamples {
            let t = i + k
            yVec[i] = series[t]
            XMat[i * numCoeffs + 0] = 1.0
            for j in 0..<k {
                XMat[i * numCoeffs + 1 + j] = series[t - 1 - j]
            }
        }
        
        return try solveLeastSquares(X: XMat, y: yVec, rows: numSamples, cols: numCoeffs)
    }
    
    private func solveLeastSquares(X: [Double], y: [Double], rows: Int, cols: Int) throws -> [Double] {
        var trans = Int8(78) // 'N'
        var r = LAPACKInteger(rows)
        var c = LAPACKInteger(cols)
        var nrhs = LAPACKInteger(1)
        var ldb = LAPACKInteger(Swift.max(rows, cols))
        var info = LAPACKInteger(0)
        
        var AColMajor = [Double](repeating: 0.0, count: rows * cols)
        for row in 0..<rows {
            for col in 0..<cols {
                AColMajor[col * rows + row] = X[row * cols + col]
            }
        }
        
        var b = [Double](repeating: 0.0, count: Int(ldb))
        for i in 0..<rows {
            b[i] = y[i]
        }
        
        var lwork = LAPACKInteger(-1)
        var workQuery = [Double](repeating: 0.0, count: 1)
        var lda = r
        dgels_wrapper(&trans, &r, &c, &nrhs, &AColMajor, &lda, &b, &ldb, &workQuery, &lwork, &info)
        
        lwork = LAPACKInteger(workQuery[0])
        var work = [Double](repeating: 0.0, count: Int(lwork))
        dgels_wrapper(&trans, &r, &c, &nrhs, &AColMajor, &lda, &b, &ldb, &work, &lwork, &info)
        
        guard info == 0 else {
            throw ForecastError.singularMatrix
        }
        
        return Array(b[0..<cols])
    }
    
    private func integrate(
        fittedDiff: [Double],
        original: [Double],
        d: Int,
        D: Int,
        s: Int,
        isForecast: Bool = false
    ) throws -> [Double] {
        var current = fittedDiff
        
        // 1. Reconstruct seasonally differenced series Z (by undoing non-seasonal differencing d times)
        var levelSeries = original
        for _ in 0..<D {
            guard levelSeries.count > s else { break }
            var diff = [Double](repeating: 0.0, count: levelSeries.count - s)
            for i in 0..<(levelSeries.count - s) {
                diff[i] = levelSeries[i + s] - levelSeries[i]
            }
            levelSeries = diff
        }
        
        for step in (0..<d).reversed() {
            var reconstructed: [Double] = []
            var subLevelSeries = levelSeries
            for _ in 0..<step {
                guard subLevelSeries.count > 1 else { break }
                var diff = [Double](repeating: 0.0, count: subLevelSeries.count - 1)
                for i in 0..<(subLevelSeries.count - 1) {
                    diff[i] = subLevelSeries[i + 1] - subLevelSeries[i]
                }
                subLevelSeries = diff
            }
            
            if isForecast {
                var lastVal = subLevelSeries.last ?? 0.0
                for diffVal in current {
                    let val = lastVal + diffVal
                    reconstructed.append(val)
                    lastVal = val
                }
            } else {
                var lastVal = subLevelSeries[0]
                reconstructed.append(lastVal)
                for diffVal in current {
                    let val = lastVal + diffVal
                    reconstructed.append(val)
                    lastVal = val
                }
            }
            current = reconstructed
        }
        
        // 2. Reconstruct original series Y (by undoing seasonal differencing D times)
        for step in (0..<D).reversed() {
            var reconstructed: [Double] = []
            var subLevelSeries = original
            for _ in 0..<step {
                guard subLevelSeries.count > s else { break }
                var diff = [Double](repeating: 0.0, count: subLevelSeries.count - s)
                for i in 0..<(subLevelSeries.count - s) {
                    diff[i] = subLevelSeries[i + s] - subLevelSeries[i]
                }
                subLevelSeries = diff
            }
            
            if isForecast {
                var history = Array(subLevelSeries.suffix(s))
                for diffVal in current {
                    let val = (history.first ?? 0.0) + diffVal
                    reconstructed.append(val)
                    history.removeFirst()
                    history.append(val)
                }
            } else {
                for i in 0..<Swift.min(s, subLevelSeries.count) {
                    reconstructed.append(subLevelSeries[i])
                }
                for i in 0..<current.count {
                    let val = reconstructed[i] + current[i]
                    reconstructed.append(val)
                }
            }
            current = reconstructed
        }
        
        return current
    }
}
