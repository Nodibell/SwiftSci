import Foundation
import Accelerate
import SwiftStats

// LAPACK functions and types are defined in LAPACK+Wrapper.swift

public enum DecompositionModel: Sendable {
    case additive
    case multiplicative
}

public enum TimeSeriesDecomposition {
    
    /// Classical additive/multiplicative decomposition of a time series.
    /// - Parameters:
    ///   - series: The input time series values.
    ///   - period: The seasonality period (e.g., 12 for monthly, 4 for quarterly).
    ///   - model: Additive or Multiplicative model.
    /// - Returns: A DecompositionResult containing trend, seasonal, residual, and original arrays.
    public static func decompose(
        series: [Double],
        period: Int,
        model: DecompositionModel = .additive
    ) throws -> DecompositionResult {
        let n = series.count
        guard n > 0 else {
            throw ForecastError.emptyTimeSeries
        }
        guard period >= 2 else {
            throw ForecastError.invalidPeriod(period)
        }
        guard n >= period * 2 else {
            throw ForecastError.insufficientLength(minimum: period * 2, got: n)
        }
        if series.contains(where: { $0.isNaN }) {
            throw ForecastError.containsNaN
        }
        if series.contains(where: { $0.isInfinite }) {
            throw ForecastError.containsInfinity
        }
        
        // 1. Compute Trend via Centered Moving Average
        var trend = [Double](repeating: Double.nan, count: n)
        
        if period % 2 == 1 {
            // Odd period: symmetric window of size p
            let half = period / 2
            for i in half..<(n - half) {
                let slice = series[i - half...i + half]
                trend[i] = vDSP.mean(slice)
            }
        } else {
            // Even period: 2x period centered moving average
            let half = period / 2
            var ma = [Double](repeating: 0.0, count: n - period + 1)
            for i in 0...(n - period) {
                let slice = series[i..<(i + period)]
                ma[i] = vDSP.mean(slice)
            }
            for i in 0..<(n - period) {
                let idx = i + half
                trend[idx] = (ma[i] + ma[i + 1]) / 2.0
            }
        }
        
        // 2. Compute Detrended Series
        var detrended = [Double](repeating: Double.nan, count: n)
        if model == .additive {
            vDSP.subtract(series, trend, result: &detrended)
        } else {
            vDSP.divide(series, trend, result: &detrended)
        }

        // 3. Compute Seasonal Component (average detrended for each period position)
        var seasonalCycle = [Double](repeating: 0.0, count: period)
        for p in 0..<period {
            var valuesAtPos: [Double] = []
            valuesAtPos.reserveCapacity(n / period + 1)
            var idx = p
            while idx < n {
                let v = detrended[idx]
                if !v.isNaN {
                    valuesAtPos.append(v)
                }
                idx += period
            }
            if !valuesAtPos.isEmpty {
                seasonalCycle[p] = vDSP.mean(valuesAtPos)
            }
        }

        // Normalize seasonal cycle so that:
        // - Additive: sum of cycle values == 0
        // - Multiplicative: mean of cycle values == 1
        let cycleMean = vDSP.mean(seasonalCycle)
        if model == .additive {
            vDSP.add(-cycleMean, seasonalCycle, result: &seasonalCycle)
        } else {
            guard cycleMean > 0 else {
                throw ForecastError.convergenceFailed(iterations: 0)
            }
            vDSP.divide(seasonalCycle, cycleMean, result: &seasonalCycle)
        }

        // Tile the seasonal cycle to fill the seasonal array
        var seasonal = [Double](repeating: 0.0, count: n)
        for i in 0..<n {
            seasonal[i] = seasonalCycle[i % period]
        }

        // 4. Compute Residual Component
        var residual = [Double](repeating: Double.nan, count: n)
        if model == .additive {
            vDSP.subtract(detrended, seasonal, result: &residual)
        } else {
            var trendSeasonal = [Double](repeating: 0.0, count: n)
            vDSP.multiply(trend, seasonal, result: &trendSeasonal)
            vDSP.divide(series, trendSeasonal, result: &residual)
        }
        
        return DecompositionResult(
            trend: trend,
            seasonal: seasonal,
            residual: residual,
            original: series
        )
    }
    
    /// Autocorrelation function (ACF) up to maxLag.
    public static func acf(series: [Double], maxLag: Int) throws -> [Double] {
        let n = series.count
        guard n > 0 else {
            throw ForecastError.emptyTimeSeries
        }
        guard maxLag >= 0 else {
            throw ForecastError.invalidAROrder(maxLag)
        }
        guard n > maxLag else {
            throw ForecastError.insufficientLength(minimum: maxLag + 1, got: n)
        }
        
        let mean = vDSP.mean(series)
        var demeaned = [Double](repeating: 0.0, count: n)
        vDSP.add(-mean, series, result: &demeaned)
        
        let sumOfSquares = vDSP.sumOfSquares(demeaned)
        guard sumOfSquares > 0 else {
            // Constant series: lag 0 is 1.0, others are 0.0
            var result = [Double](repeating: 0.0, count: maxLag + 1)
            result[0] = 1.0
            return result
        }
        
        var result = [Double](repeating: 0.0, count: maxLag + 1)
        result[0] = 1.0
        
        for lag in 1...maxLag {
            var sum = 0.0
            for i in 0..<(n - lag) {
                sum += demeaned[i] * demeaned[i + lag]
            }
            result[lag] = sum / sumOfSquares
        }
        
        return result
    }
    
    /// Partial Autocorrelation function (PACF) up to maxLag using Yule-Walker equations.
    public static func pacf(series: [Double], maxLag: Int) throws -> [Double] {
        let n = series.count
        guard n > 0 else {
            throw ForecastError.emptyTimeSeries
        }
        guard maxLag >= 0 else {
            throw ForecastError.invalidAROrder(maxLag)
        }
        guard n > maxLag else {
            throw ForecastError.insufficientLength(minimum: maxLag + 1, got: n)
        }
        
        let r = try acf(series: series, maxLag: maxLag)
        var result = [Double](repeating: 0.0, count: maxLag + 1)
        result[0] = 1.0
        
        if maxLag == 0 {
            return result
        }
        
        result[1] = r[1]
        if maxLag == 1 {
            return result
        }
        
        for k in 2...maxLag {
            // Solve Yule-Walker system R_k * phi = r_k
            // R_k is k x k Toeplitz matrix filled with r[0] ... r[k-1]
            // r_k is vector of size k filled with r[1] ... r[k]
            // R is a symmetric Toeplitz matrix. We populate RColMajor explicitly
            // in column-major layout to satisfy the LAPACK requirement.
            var RColMajor = [Double](repeating: 0.0, count: k * k)
            var rhs = [Double](repeating: 0.0, count: k)
            
            for i in 0..<k {
                rhs[i] = r[i + 1]
                for j in 0..<k {
                    let lag = abs(i - j)
                    RColMajor[j * k + i] = r[lag]
                }
            }
            
            // Solve using dgesv_wrapper
            var dim = LAPACKInteger(k)
            var nrhs = LAPACKInteger(1)
            var ipiv = [LAPACKInteger](repeating: 0, count: k)
            var info = LAPACKInteger(0)
            var lda = dim
            var ldb = dim
            
            dgesv_wrapper(&dim, &nrhs, &RColMajor, &lda, &ipiv, &rhs, &ldb, &info)
            
            if info == 0 {
                result[k] = rhs[k - 1]
            } else {
                result[k] = 0.0 // Default fallback on numerical issues
            }
        }
        
        return result
    }
    
    /// Augmented Dickey-Fuller (ADF) stationarity test (lags = maxLag, with constant).
    /// Returns: (statistic, pValue)
    public static func adfTest(series: [Double], maxLag: Int = 1) throws -> (statistic: Double, pValue: Double) {
        let n = series.count
        guard n > maxLag + 2 else {
            throw ForecastError.insufficientLength(minimum: maxLag + 3, got: n)
        }
        
        // 1. Difference the series once: dy = y_t - y_{t-1}
        var dy = [Double](repeating: 0.0, count: n - 1)
        for i in 0..<(n - 1) {
            dy[i] = series[i + 1] - series[i]
        }
        
        // 2. Setup regression targets & design matrix
        // dy_t = alpha + beta * y_{t-1} + delta_1 * dy_{t-1} + ... + delta_k * dy_{t-k}
        // Samples run from t = maxLag + 1 to n - 1 (0-indexed: index maxLag to n - 2 in dy)
        let numSamples = n - 1 - maxLag
        let numCoeffs = 2 + maxLag // Intercept, y_{t-1}, plus maxLag lagged dy
        
        var yVec = [Double](repeating: 0.0, count: numSamples)
        var XMat = [Double](repeating: 0.0, count: numSamples * numCoeffs)
        
        for i in 0..<numSamples {
            let t = i + maxLag
            yVec[i] = dy[t]
            
            // Column 0: Intercept
            XMat[i * numCoeffs + 0] = 1.0
            
            // Column 1: lagged level y_{t-1}
            XMat[i * numCoeffs + 1] = series[t]
            
            // Column 2 to 2+maxLag-1: lagged differences dy_{t-1} ... dy_{t-maxLag}
            for lag in 1...maxLag {
                XMat[i * numCoeffs + 1 + lag] = dy[t - lag]
            }
        }
        
        // 3. Solve OLS via LAPACK dgels_
        // dgels solves min ||b - A*x||
        // Note: A is overwritten by QR/LQ, b is overwritten by solution
        var trans = Int8(78) // 'N'
        var rows = LAPACKInteger(numSamples)
        var cols = LAPACKInteger(numCoeffs)
        var nrhs = LAPACKInteger(1)
        var lda = rows
        var ldb = LAPACKInteger(Swift.max(numSamples, numCoeffs))
        var info = LAPACKInteger(0)
        

        // dgels expects column-major. Let's transpose XMat to column-major.
        var AColMajor = [Double](repeating: 0.0, count: numSamples * numCoeffs)
        for r in 0..<numSamples {
            for c in 0..<numCoeffs {
                AColMajor[c * numSamples + r] = XMat[r * numCoeffs + c]
            }
        }
        
        var b = [Double](repeating: 0.0, count: Int(ldb))
        for i in 0..<numSamples {
            b[i] = yVec[i]
        }
        
        var lwork = LAPACKInteger(-1)
        var workQuery = [Double](repeating: 0.0, count: 1)
        dgels_wrapper(&trans, &rows, &cols, &nrhs, &AColMajor, &lda, &b, &ldb, &workQuery, &lwork, &info)
        
        lwork = LAPACKInteger(workQuery[0])
        var work = [Double](repeating: 0.0, count: Int(lwork))
        dgels_wrapper(&trans, &rows, &cols, &nrhs, &AColMajor, &lda, &b, &ldb, &work, &lwork, &info)
        
        guard info == 0 else {
            throw ForecastError.singularMatrix
        }
        
        // Beta coefficients are in b[0...numCoeffs-1]
        let beta = Array(b[0..<numCoeffs])
        
        // 4. Compute residuals and standard error for beta[1] (y_{t-1})
        var residualsSumOfSquares = 0.0
        for i in 0..<numSamples {
            var prediction = 0.0
            for c in 0..<numCoeffs {
                prediction += XMat[i * numCoeffs + c] * beta[c]
            }
            let diff = yVec[i] - prediction
            residualsSumOfSquares += diff * diff
        }
        
        let df = Double(numSamples - numCoeffs)
        guard df > 0 else {
            throw ForecastError.insufficientLength(minimum: numCoeffs + 1, got: numSamples)
        }
        let residualVariance = residualsSumOfSquares / df
        
        // Compute (X^T * X)^-1 to get the variance of the coefficients
        var XtX = [Double](repeating: 0.0, count: numCoeffs * numCoeffs)
        for i in 0..<numCoeffs {
            for j in 0..<numCoeffs {
                var sum = 0.0
                for k in 0..<numSamples {
                    sum += XMat[k * numCoeffs + i] * XMat[k * numCoeffs + j]
                }
                XtX[i * numCoeffs + j] = sum
            }
        }
        
        // Invert XtX using dpotrf_wrapper and dpotri_wrapper (for symmetric positive definite matrices)
        var uplo = Int8(85) // 'U'
        var nInt = LAPACKInteger(numCoeffs)
        var ldaMat = nInt
        var infoInv = LAPACKInteger(0)
        dpotrf_wrapper(&uplo, &nInt, &XtX, &ldaMat, &infoInv)
        
        guard infoInv == 0 else {
            throw ForecastError.singularMatrix
        }
        
        dpotri_wrapper(&uplo, &nInt, &XtX, &ldaMat, &infoInv)
        
        let seBeta1 = sqrt(residualVariance * XtX[1 * numCoeffs + 1])
        
        guard seBeta1 > 0 else {
            throw ForecastError.singularMatrix
        }
        
        let tStat = beta[1] / seBeta1
        
        // 5. Approximate p-value based on MacKinnon's tables for ADF test with constant
        // For N=infinity, critical values are: 1%: -3.43, 5%: -2.86, 10%: -2.57
        // Simple interpolation/logistic function for ADF p-value approximation:
        let pVal: Double
        if tStat <= -3.43 {
            pVal = 0.01 * exp((tStat - (-3.43)) * 4.0)
        } else if tStat <= -2.86 {
            // Linear interpolation between 1% and 5%
            pVal = 0.01 + (tStat - (-3.43)) / (-2.86 - (-3.43)) * 0.04
        } else if tStat <= -2.57 {
            // Linear interpolation between 5% and 10%
            pVal = 0.05 + (tStat - (-2.86)) / (-2.57 - (-2.86)) * 0.05
        } else {
            // Above 10%, map to [0.1, 1.0] using sigmoid-like curve
            let diff = tStat - (-2.57)
            pVal = 0.10 + 0.90 * (1.0 - exp(-diff * 1.5))
        }
        
        return (tStat, Swift.max(0.0, Swift.min(1.0, pVal)))
    }
}
