import Foundation
import Accelerate
import SwiftStats

#if ACCELERATE_NEW_LAPACK
// Int32 version
#else
// __CLPK_integer version
#endif

public actor ARIMAModel {
    public let order: (p: Int, d: Int, q: Int)
    
    private var arCoefficients: [Double] = []
    private var maCoefficients: [Double] = []
    private var intercept: Double = 0.0
    
    private var exogCoefficients: [Double] = []
    private var regIntercept: Double = 0.0
    private var fittedExog: [[Double]]? = nil
    private var hasExog = false
    
    private var series: [Double] = []
    private var diffSeries: [Double] = []
    private var fittedValues: [Double] = []
    private var residuals: [Double] = []
    private var isFitted = false
    
    public init(p: Int, d: Int, q: Int) throws {
        guard p >= 0 else { throw ForecastError.invalidAROrder(p) }
        guard d >= 0 else { throw ForecastError.invalidDifferencing(d) }
        guard q >= 0 else { throw ForecastError.invalidMAOrder(q) }
        self.order = (p, d, q)
    }
    
    /// Fits the ARIMA model on the series using Hannan-Rissanen conditional OLS.
    public func fit(series: [Double], exog: [[Double]]? = nil) throws {
        let n = series.count
        guard n > 0 else { throw ForecastError.emptyTimeSeries }
        
        let minLen = order.p + order.d + order.q + 2
        guard n >= minLen else {
            throw ForecastError.insufficientLength(minimum: minLen, got: n)
        }
        if series.contains(where: { $0.isNaN }) { throw ForecastError.containsNaN }
        if series.contains(where: { $0.isInfinite }) { throw ForecastError.containsInfinity }
        
        var arimaTargetSeries = series
        
        if let exog = exog {
            guard exog.count == n else {
                throw ForecastError.matrixDimensionMismatch(expectedRows: n, expectedCols: 1, gotRows: exog.count, gotCols: 1)
            }
            let numExog = exog[0].count
            guard numExog > 0 else {
                throw ForecastError.matrixDimensionMismatch(expectedRows: n, expectedCols: 1, gotRows: n, gotCols: 0)
            }
            for row in exog {
                guard row.count == numExog else {
                    throw ForecastError.matrixDimensionMismatch(expectedRows: n, expectedCols: numExog, gotRows: n, gotCols: row.count)
                }
            }
            
            // Fit OLS regression: series = exog * beta + intercept
            let numCoeffsReg = numExog + 1
            var XMatReg = [Double](repeating: 0.0, count: n * numCoeffsReg)
            for i in 0..<n {
                XMatReg[i * numCoeffsReg + 0] = 1.0 // intercept
                for j in 0..<numExog {
                    XMatReg[i * numCoeffsReg + 1 + j] = exog[i][j]
                }
            }
            
            let coeffsReg = try solveLeastSquares(X: XMatReg, y: series, rows: n, cols: numCoeffsReg)
            self.regIntercept = coeffsReg[0]
            self.exogCoefficients = Array(coeffsReg[1...numExog])
            self.fittedExog = exog
            self.hasExog = true
            
            // Compute residuals (which will be fitted by ARIMA)
            var residualsReg = [Double](repeating: 0.0, count: n)
            for i in 0..<n {
                var regVal = regIntercept
                for j in 0..<numExog {
                    regVal += exogCoefficients[j] * exog[i][j]
                }
                residualsReg[i] = series[i] - regVal
            }
            arimaTargetSeries = residualsReg
        } else {
            self.hasExog = false
            self.regIntercept = 0.0
            self.exogCoefficients = []
            self.fittedExog = nil
        }
        
        self.series = arimaTargetSeries
        // 1. Difference the series d times
        var current = arimaTargetSeries
        for _ in 0..<order.d {
            var diff = [Double](repeating: 0.0, count: current.count - 1)
            for i in 0..<(current.count - 1) {
                diff[i] = current[i + 1] - current[i]
            }
            current = diff
        }
        self.diffSeries = current
        
        let nDiff = diffSeries.count
        
        // Handle trivial ARIMA(0, 0, 0)
        if order.p == 0 && order.q == 0 {
            self.intercept = vDSP.mean(diffSeries)
            self.arCoefficients = []
            self.maCoefficients = []
            self.fittedValues = [Double](repeating: intercept, count: nDiff)
            self.residuals = [Double](repeating: 0.0, count: nDiff)
            for i in 0..<nDiff {
                self.residuals[i] = diffSeries[i] - intercept
            }
            
            // Reconstruct original scale fitted values
            self.fittedValues = try integrate(fittedDiff: self.fittedValues, original: series, d: order.d)
            self.isFitted = true
            return
        }
        
        // 2. High-order AR model to estimate initial residuals (Hannan-Rissanen Step 1)
        // High-order k = max(p + q, 4)
        let k = Swift.max(order.p + order.q, 4)
        guard nDiff > k + 2 else {
            throw ForecastError.insufficientLength(minimum: k + order.d + 3, got: n)
        }
        
        let highArCoeffs = try fitAR(series: diffSeries, order: k)
        
        // Compute residuals from this high-order AR model
        var estResiduals = [Double](repeating: 0.0, count: nDiff)
        for t in k..<nDiff {
            var val = highArCoeffs[0] // intercept
            for i in 0..<k {
                val += highArCoeffs[1 + i] * diffSeries[t - 1 - i]
            }
            estResiduals[t] = diffSeries[t] - val
        }
        
        // 3. Conditional OLS for ARIMA(p, 0, q) on diffSeries (Hannan-Rissanen Step 2)
        let startIdx = Swift.max(order.p, order.q)
        let numSamples = nDiff - startIdx
        let numCoeffs = 1 + order.p + order.q
        
        var yVec = [Double](repeating: 0.0, count: numSamples)
        var XMat = [Double](repeating: 0.0, count: numSamples * numCoeffs)
        
        for i in 0..<numSamples {
            let t = i + startIdx
            yVec[i] = diffSeries[t]
            
            // Intercept
            XMat[i * numCoeffs + 0] = 1.0
            
            // AR terms
            for j in 0..<order.p {
                XMat[i * numCoeffs + 1 + j] = diffSeries[t - 1 - j]
            }
            
            // MA terms
            for j in 0..<order.q {
                XMat[i * numCoeffs + 1 + order.p + j] = estResiduals[t - 1 - j]
            }
        }
        
        let coeffs = try solveLeastSquares(X: XMat, y: yVec, rows: numSamples, cols: numCoeffs)
        
        self.intercept = coeffs[0]
        if order.p > 0 {
            self.arCoefficients = Array(coeffs[1...order.p])
        } else {
            self.arCoefficients = []
        }
        if order.q > 0 {
            self.maCoefficients = Array(coeffs[(1 + order.p)...(order.p + order.q)])
        } else {
            self.maCoefficients = []
        }
        
        // 4. Compute in-sample fitted values and residuals on differenced series
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
                for j in 0..<order.q {
                    val += maCoefficients[j] * diffResiduals[t - 1 - j]
                }
                diffFitted[t] = val
                diffResiduals[t] = diffSeries[t] - val
            }
        }
        
        self.residuals = diffResiduals
        self.fittedValues = try integrate(fittedDiff: diffFitted, original: self.series, d: order.d)
        self.isFitted = true
    }
    
    /// Forecasts horizon steps ahead.
    public func forecast(horizon: Int, exog: [[Double]]? = nil) throws -> ARIMAResult {
        guard isFitted else {
            throw ForecastError.notFitted
        }
        guard horizon >= 1 else {
            throw ForecastError.invalidHorizon(horizon)
        }
        
        if hasExog {
            guard let exog = exog else {
                throw ForecastError.matrixDimensionMismatch(expectedRows: horizon, expectedCols: exogCoefficients.count, gotRows: 0, gotCols: 0)
            }
            guard exog.count == horizon else {
                throw ForecastError.matrixDimensionMismatch(expectedRows: horizon, expectedCols: exogCoefficients.count, gotRows: exog.count, gotCols: exog[0].count)
            }
            for row in exog {
                guard row.count == exogCoefficients.count else {
                    throw ForecastError.matrixDimensionMismatch(expectedRows: horizon, expectedCols: exogCoefficients.count, gotRows: horizon, gotCols: row.count)
                }
            }
        }
        
        let nDiff = diffSeries.count
        var diffPreds = [Double](repeating: 0.0, count: horizon)
        var allDiffSeries = diffSeries
        var allResiduals = residuals
        
        for h in 0..<horizon {
            let t = nDiff + h
            var val = intercept
            
            // AR part
            for j in 0..<order.p {
                let lagIdx = t - 1 - j
                let lagVal = lagIdx < nDiff ? allDiffSeries[lagIdx] : diffPreds[lagIdx - nDiff]
                val += arCoefficients[j] * lagVal
            }
            
            // MA part (residuals are 0 for out-of-sample)
            for j in 0..<order.q {
                let lagIdx = t - 1 - j
                let lagRes = lagIdx < nDiff ? allResiduals[lagIdx] : 0.0
                val += maCoefficients[j] * lagRes
            }
            
            diffPreds[h] = val
            allDiffSeries.append(val)
            allResiduals.append(0.0) // forecast residual is 0
        }
        
        // Integrate forecasts back to original scale
        let integratedForecastsResiduals = try integrate(fittedDiff: diffPreds, original: series, d: order.d, isForecast: true)
        
        // Calculate final predictions (adding exog part back if needed)
        var finalPredictions = integratedForecastsResiduals
        if hasExog, let exog = exog {
            for h in 0..<horizon {
                var regVal = regIntercept
                for j in 0..<exogCoefficients.count {
                    regVal += exogCoefficients[j] * exog[h][j]
                }
                finalPredictions[h] = integratedForecastsResiduals[h] + regVal
            }
        }
        
        // Calculate metrics
        let nOriginal = series.count
        var origFitted = Array(fittedValues.prefix(nOriginal))
        while origFitted.count < nOriginal {
            origFitted.append(series.last ?? 0.0)
        }
        
        var originalResiduals = [Double](repeating: 0.0, count: nOriginal)
        vDSP.subtract(series, origFitted, result: &originalResiduals)
        
        let startIdx = Swift.max(order.p, order.q)
        let validResiduals = Array(residuals[startIdx...])
        let validCount = Double(validResiduals.count)
        
        let mse = validCount > 0 ? (vDSP.sumOfSquares(validResiduals) / validCount) : 0.0
        let mae = validCount > 0 ? (validResiduals.reduce(0.0) { $0 + abs($1) } / validCount) : 0.0
        
        // Compute AIC = 2k - 2ln(L)
        // Log-likelihood approximation: -N/2 * (1 + ln(2*pi*MSE))
        let k = Double(1 + order.p + order.q + (hasExog ? exogCoefficients.count : 0))
        let aic = validCount > 0 ? (2.0 * k + validCount * log(mse > 0 ? mse : 1e-15) + validCount * (1.0 + log(2.0 * Double.pi))) : 0.0
        
        let forecastResult = ForecastResult(
            predictions: finalPredictions,
            lowerBound: nil,
            upperBound: nil,
            fittedValues: fittedValues,
            residuals: residuals,
            aic: aic,
            mse: mse,
            mae: mae
        )
        
        return ARIMAResult(
            order: order,
            arCoefficients: arCoefficients,
            maCoefficients: maCoefficients,
            intercept: intercept,
            exogCoefficients: exogCoefficients,
            forecast: forecastResult
        )
    }
    
    public func aic() throws -> Double {
        guard isFitted else { throw ForecastError.notFitted }
        let mse = vDSP.sumOfSquares(residuals) / Double(residuals.count)
        let k = Double(1 + order.p + order.q)
        let n = Double(residuals.count)
        return 2.0 * k + n * log(mse > 0 ? mse : 1e-15) + n * (1.0 + log(2.0 * Double.pi))
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
        
        // Transpose to column-major
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
    
    private func integrate(fittedDiff: [Double], original: [Double], d: Int, isForecast: Bool = false) throws -> [Double] {
        if d == 0 {
            return fittedDiff
        }
        
        var current = fittedDiff
        
        // Reconstruct step-by-step
        // If we differenced d times:
        // Let's say d = 1. original is y_0, ..., y_{n-1}. diffSeries is dy_1, ..., dy_{n-1} (dy_i = y_i - y_{i-1}).
        // If isForecast is true:
        //   fittedDiff are predictions for dy_n, dy_{n+1}, ...
        //   We use y_{n-1} as the starting point: y_n = y_{n-1} + dy_n, y_{n+1} = y_n + dy_{n+1}.
        // If isForecast is false:
        //   fittedDiff are fitted values for dy_0, ..., dy_{n-1}.
        //   We reconstruct starting from y_0.
        
        for step in (0..<d).reversed() {
            var reconstructed: [Double] = []
            if isForecast {
                // Determine the correct starting value from the original series
                // For order.d - 1 - step differencing level
                var levelSeries = original
                for _ in 0..<step {
                    var diff = [Double](repeating: 0.0, count: levelSeries.count - 1)
                    for i in 0..<(levelSeries.count - 1) {
                        diff[i] = levelSeries[i + 1] - levelSeries[i]
                    }
                    levelSeries = diff
                }
                
                var lastVal = levelSeries.last ?? 0.0
                for diffVal in current {
                    let val = lastVal + diffVal
                    reconstructed.append(val)
                    lastVal = val
                }
            } else {
                // Fit mode: reconstruct full series of length `n - step`
                var levelSeries = original
                for _ in 0..<step {
                    var diff = [Double](repeating: 0.0, count: levelSeries.count - 1)
                    for i in 0..<(levelSeries.count - 1) {
                        diff[i] = levelSeries[i + 1] - levelSeries[i]
                    }
                    levelSeries = diff
                }
                
                // Reconstruct from levelSeries[0]
                var lastVal = levelSeries[0]
                reconstructed.append(lastVal)
                for diffVal in current {
                    let val = lastVal + diffVal
                    reconstructed.append(val)
                    lastVal = val
                }
            }
            current = reconstructed
        }
        
        return current
    }
}
