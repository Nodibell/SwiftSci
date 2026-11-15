import Foundation
import Accelerate
import SwiftStats

/// Seasonal AutoRegressive Integrated Moving Average (SARIMA) time-series forecasting model actor.
///
/// Implements a multiplicative SARIMA-style model (p, d, q) x (P, D, Q)_s with
/// conditional least-squares estimation and polynomial cross-term refinement.
/// Parameter estimation utilizes high-order AR approximation followed by two-stage conditional least-squares
/// polynomial refinement rather than full state-space exact maximum likelihood estimation.
public actor SARIMAModel {
    /// The non-seasonal order (p, d, q).
    public let order: (p: Int, d: Int, q: Int)
    /// The seasonal order (P, D, Q, s).
    public let seasonalOrder: (P: Int, D: Int, Q: Int, s: Int)

    public private(set) var arCoefficients: [Double] = []
    public private(set) var maCoefficients: [Double] = []
    public private(set) var seasonalArCoefficients: [Double] = []
    public private(set) var seasonalMaCoefficients: [Double] = []
    public private(set) var intercept: Double = 0.0

    private var series: [Double] = []
    private var diffSeries: [Double] = []
    /// One-step-ahead in-sample fitted values on the original data scale (`count == series.count`).
    /// Initial boundary indices `0..<(order.d + seasonalOrder.D * seasonalOrder.s)` retain seed observations;
    /// for post-warmup indices `t`, `fittedValues[t] == series[t] - residuals[t - warmUp]`.
    public private(set) var fittedValues: [Double] = []
    /// One-step-ahead conditional innovations (`count == series.count - (order.d + seasonalOrder.D * seasonalOrder.s)`),
    /// aligned so that `residuals[t - warmUp] == series[t] - fittedValues[t]` for `t >= warmUp`.
    public private(set) var residuals: [Double] = []
    public private(set) var isFitted: Bool = false

    /// Creates a new SARIMA model instance with specified non-seasonal and seasonal orders.
    ///
    /// - Parameters:
    ///   - p: Autoregressive order \(p \ge 0\).
    ///   - d: Non-seasonal differencing order \(d \ge 0\).
    ///   - q: Moving average order \(q \ge 0\).
    ///   - P: Seasonal autoregressive order \(P \ge 0\).
    ///   - D: Seasonal differencing order \(D \ge 0\).
    ///   - Q: Seasonal moving average order \(Q \ge 0\).
    ///   - s: Seasonal period length \(s \ge 1\).
    /// - Throws: `ForecastError` if orders are negative or period is less than 1.
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

    private func resetFittedState() {
        self.isFitted = false
        self.arCoefficients = []
        self.maCoefficients = []
        self.seasonalArCoefficients = []
        self.seasonalMaCoefficients = []
        self.intercept = 0.0
        self.series = []
        self.diffSeries = []
        self.fittedValues = []
        self.residuals = []
    }

    /// Fits the SARIMA model to the given time series using two-stage conditional least squares
    /// with multiplicative polynomial cross-term refinement.
    ///
    /// - Parameter series: Array of historical numerical observations.
    /// - Throws: `ForecastError` if series contains NaNs/Infs, is empty, has insufficient length,
    ///   yields non-invertible moving-average polynomials, or produces non-finite residuals/fitted values.
    public func fit(series: [Double]) throws {
        resetFittedState()
        do {
            try performFit(series: series)
        } catch {
            resetFittedState()
            throw error
        }
    }

    private func performFit(series: [Double]) throws {
        let n = series.count
        guard n > 0 else { throw ForecastError.emptyTimeSeries }
        try ForecastValidation.ensureFinite(series)

        let s = seasonalOrder.s
        let maxLag = Swift.max(order.p + seasonalOrder.P * s, order.q + seasonalOrder.Q * s)
        let minLen = Swift.max(maxLag + order.d + seasonalOrder.D * s + 3, 6)
        guard n >= minLen else {
            throw ForecastError.insufficientLength(minimum: minLen, got: n)
        }

        self.series = series

        // 1. Difference the series: seasonal D times, then regular d times
        var current = series
        for _ in 0..<seasonalOrder.D {
            guard current.count > s else { break }
            var diff = [Double](repeating: 0.0, count: current.count - s)
            for i in 0..<(current.count - s) {
                diff[i] = current[i + s] - current[i]
            }
            current = diff
        }
        for _ in 0..<order.d {
            guard current.count > 1 else { break }
            var diff = [Double](repeating: 0.0, count: current.count - 1)
            for i in 0..<(current.count - 1) {
                diff[i] = current[i + 1] - current[i]
            }
            current = diff
        }
        try ForecastValidation.ensureFinite(current)
        self.diffSeries = current

        let nDiff = diffSeries.count

        // Zero-variance guard: if differenced series is constant, fit trivial constant model
        let minDiff = diffSeries.min() ?? 0.0
        let maxDiff = diffSeries.max() ?? 0.0
        if abs(maxDiff - minDiff) < 1e-12 {
            try ForecastValidation.ensureFinite(minDiff)
            let diffFitted = [Double](repeating: minDiff, count: nDiff)
            let diffResiduals = [Double](repeating: 0.0, count: nDiff)
            let integratedFitted = try integrate(fittedDiff: diffFitted, original: series, d: order.d, D: seasonalOrder.D, s: s)
            try ForecastValidation.ensureFinite(integratedFitted)
            self.intercept = minDiff
            self.arCoefficients = [Double](repeating: 0.0, count: order.p)
            self.seasonalArCoefficients = [Double](repeating: 0.0, count: seasonalOrder.P)
            self.maCoefficients = [Double](repeating: 0.0, count: order.q)
            self.seasonalMaCoefficients = [Double](repeating: 0.0, count: seasonalOrder.Q)
            self.residuals = diffResiduals
            self.fittedValues = integratedFitted
            self.isFitted = true
            return
        }

        // 2. High-order AR approximation to estimate initial innovations / residuals
        let k = Swift.min(Swift.max(maxLag, 4), Swift.max(nDiff / 4, 2))
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
        try ForecastValidation.ensureFinite(estResiduals)

        // 3. Conditional Least Squares with Multiplicative Polynomial Terms
        let startIdx = Swift.max(maxLag, 1)
        let numSamples = nDiff - startIdx
        guard numSamples > 0 else {
            throw ForecastError.insufficientLength(minimum: minLen, got: n)
        }

        let numCoeffs = 1 + order.p + seasonalOrder.P + order.q + seasonalOrder.Q

        var yVec = [Double](repeating: 0.0, count: numSamples)
        var XMat = [Double](repeating: 0.0, count: numSamples * numCoeffs)

        for i in 0..<numSamples {
            let t = i + startIdx
            yVec[i] = diffSeries[t]
            XMat[i * numCoeffs + 0] = 1.0 // intercept

            var colIdx = 1
            // Non-seasonal AR lags: w[t - 1 - j]
            for j in 0..<order.p {
                XMat[i * numCoeffs + colIdx] = diffSeries[t - 1 - j]
                colIdx += 1
            }
            // Seasonal AR lags: w[t - (j + 1) * s]
            for j in 0..<seasonalOrder.P {
                XMat[i * numCoeffs + colIdx] = diffSeries[t - (j + 1) * s]
                colIdx += 1
            }
            // Non-seasonal MA lags: e[t - 1 - j]
            for j in 0..<order.q {
                XMat[i * numCoeffs + colIdx] = estResiduals[t - 1 - j]
                colIdx += 1
            }
            // Seasonal MA lags: e[t - (j + 1) * s]
            for j in 0..<seasonalOrder.Q {
                XMat[i * numCoeffs + colIdx] = estResiduals[t - (j + 1) * s]
                colIdx += 1
            }
        }

        // Initial parameter estimates
        var coeffs = try solveLeastSquares(X: XMat, y: yVec, rows: numSamples, cols: numCoeffs)

        func extractCoefficients(_ raw: [Double]) -> (
            intercept: Double,
            ar: [Double],
            sar: [Double],
            ma: [Double],
            sma: [Double]
        ) {
            let ic = raw[0]
            var idx = 1
            let ar = order.p > 0 ? Array(raw[idx..<(idx + order.p)]) : []
            idx += order.p
            let sar = seasonalOrder.P > 0 ? Array(raw[idx..<(idx + seasonalOrder.P)]) : []
            idx += seasonalOrder.P
            let ma = order.q > 0 ? Array(raw[idx..<(idx + order.q)]) : []
            idx += order.q
            let sma = seasonalOrder.Q > 0 ? Array(raw[idx..<(idx + seasonalOrder.Q)]) : []
            return (ic, ar, sar, ma, sma)
        }

        var (curIntercept, curAR, curSAR, curMA, curSMA) = extractCoefficients(coeffs)

        // Refine with multiplicative polynomial cross-terms if both non-seasonal and seasonal orders exist
        let hasCrossTerms = (order.p > 0 && seasonalOrder.P > 0) || (order.q > 0 && seasonalOrder.Q > 0)
        if hasCrossTerms {
            let refineIterations = 2
            for _ in 0..<refineIterations {
                var yAdjusted = [Double](repeating: 0.0, count: numSamples)
                for i in 0..<numSamples {
                    let t = i + startIdx
                    var crossAR = 0.0
                    for j in 0..<order.p {
                        for J in 0..<seasonalOrder.P {
                            let lagIdx = t - 1 - j - (J + 1) * s
                            if lagIdx >= 0 {
                                crossAR += curAR[j] * curSAR[J] * diffSeries[lagIdx]
                            }
                        }
                    }
                    var crossMA = 0.0
                    for j in 0..<order.q {
                        for J in 0..<seasonalOrder.Q {
                            let lagIdx = t - 1 - j - (J + 1) * s
                            if lagIdx >= 0 {
                                crossMA += curMA[j] * curSMA[J] * estResiduals[lagIdx]
                            }
                        }
                    }
                    // Adjusted target: subtract cross term influence
                    // Since w_t = linear_terms - crossAR + crossMA,
                    // linear_terms = w_t + crossAR - crossMA
                    yAdjusted[i] = diffSeries[t] + crossAR - crossMA
                }

                if yAdjusted.allSatisfy(\.isFinite),
                   let refined = try? solveLeastSquares(X: XMat, y: yAdjusted, rows: numSamples, cols: numCoeffs) {
                    coeffs = refined
                    (curIntercept, curAR, curSAR, curMA, curSMA) = extractCoefficients(coeffs)
                }
            }
        }

        try ForecastValidation.ensureFinite(curIntercept)
        try ForecastValidation.ensureFinite(curAR)
        try ForecastValidation.ensureFinite(curSAR)
        try ForecastValidation.ensureFinite(curMA)
        try ForecastValidation.ensureFinite(curSMA)
        try ForecastValidation.ensureMAInvertible(ma: curMA, seasonalMA: curSMA)

        // 4. In-sample multi-lag multiplicative recursion to produce fitted values and residuals
        var diffFitted = [Double](repeating: 0.0, count: nDiff)
        var diffResiduals = [Double](repeating: 0.0, count: nDiff)

        for t in 0..<nDiff {
            if t < startIdx {
                diffFitted[t] = diffSeries[t]
                diffResiduals[t] = 0.0
            } else {
                let pred = computeMultiplicativePrediction(
                    historyW: diffSeries,
                    t: t,
                    historyE: diffResiduals,
                    p: order.p,
                    P: seasonalOrder.P,
                    q: order.q,
                    Q: seasonalOrder.Q,
                    s: s,
                    ar: curAR,
                    sar: curSAR,
                    ma: curMA,
                    sma: curSMA,
                    intercept: curIntercept
                )
                diffFitted[t] = pred
                diffResiduals[t] = diffSeries[t] - pred
            }
        }

        try ForecastValidation.ensureFinite(diffFitted)
        try ForecastValidation.ensureFinite(diffResiduals)
        let integratedFitted = try integrate(fittedDiff: diffFitted, original: series, d: order.d, D: seasonalOrder.D, s: s)
        try ForecastValidation.ensureFinite(integratedFitted)

        self.intercept = curIntercept
        self.arCoefficients = curAR
        self.seasonalArCoefficients = curSAR
        self.maCoefficients = curMA
        self.seasonalMaCoefficients = curSMA
        self.residuals = diffResiduals
        self.fittedValues = integratedFitted
        self.isFitted = true
    }

    /// Forecasts `steps` periods ahead using the full multiplicative SARIMA operator.
    ///
    /// - Parameter steps: Number of future time steps to forecast. Must be at least 1.
    /// - Throws: `ForecastError.notFitted` if unfitted, `ForecastError.invalidHorizon` if steps < 1,
    ///   or `ForecastError.containsNaN` / `ForecastError.containsInfinity` if the extrapolation becomes non-finite.
    /// - Returns: Array of forecasted future values in the original series scale.
    public func forecast(steps: Int) throws -> [Double] {
        guard isFitted else { throw ForecastError.notFitted }
        guard steps >= 1 else { throw ForecastError.invalidHorizon(steps) }
        try ForecastValidation.ensureFinite(residuals)

        let s = seasonalOrder.s
        var fcDiff = [Double](repeating: 0.0, count: steps)
        var historyDiff = diffSeries
        var historyResiduals = residuals

        for step in 0..<steps {
            let t = historyDiff.count
            let val = computeMultiplicativePrediction(
                historyW: historyDiff,
                t: t,
                historyE: historyResiduals,
                p: order.p,
                P: seasonalOrder.P,
                q: order.q,
                Q: seasonalOrder.Q,
                s: s,
                ar: arCoefficients,
                sar: seasonalArCoefficients,
                ma: maCoefficients,
                sma: seasonalMaCoefficients,
                intercept: intercept
            )
            fcDiff[step] = val
            historyDiff.append(val)
            historyResiduals.append(0.0) // future residuals are zero in expectation
        }

        try ForecastValidation.ensureFinite(fcDiff)
        let integrated = try integrate(fittedDiff: fcDiff, original: series, d: order.d, D: seasonalOrder.D, s: s, isForecast: true)
        try ForecastValidation.ensureFinite(integrated)
        return integrated
    }

    /// Computes the Akaike Information Criterion (AIC) for the fitted SARIMA model.
    ///
    /// - Throws: `ForecastError.notFitted` if model has not been fitted, or
    ///   `ForecastError.containsNaN` / `ForecastError.containsInfinity` if residuals or criterion are non-finite.
    /// - Returns: AIC score as a `Double`.
    public func aic() throws -> Double {
        let (mse, k, n) = try informationCriterionInputs()
        let score = 2.0 * k + n * log(mse > 0 ? mse : 1e-15) + n * (1.0 + log(2.0 * Double.pi))
        try ForecastValidation.ensureFinite(score)
        return score
    }

    /// Computes the Bayesian Information Criterion (BIC) for the fitted SARIMA model.
    ///
    /// - Throws: `ForecastError.notFitted` if model has not been fitted, or
    ///   `ForecastError.containsNaN` / `ForecastError.containsInfinity` if residuals or criterion are non-finite.
    /// - Returns: BIC score as a `Double`.
    public func bic() throws -> Double {
        let (mse, k, n) = try informationCriterionInputs()
        let score = k * log(n) + n * log(mse > 0 ? mse : 1e-15) + n * (1.0 + log(2.0 * Double.pi))
        try ForecastValidation.ensureFinite(score)
        return score
    }

    private func informationCriterionInputs() throws -> (mse: Double, k: Double, n: Double) {
        guard isFitted else { throw ForecastError.notFitted }
        try ForecastValidation.ensureFinite(residuals)
        guard !residuals.isEmpty else { throw ForecastError.insufficientLength(minimum: 1, got: 0) }
        let n = Double(residuals.count)
        let mse = vDSP.sumOfSquares(residuals) / n
        try ForecastValidation.ensureFinite(mse)
        let k = Double(1 + order.p + order.q + seasonalOrder.P + seasonalOrder.Q)
        return (mse, k, n)
    }

    // MARK: - Private helper methods

    /// Computes recursive one-step prediction applying the multiplicative SARIMA polynomial:
    /// \(\phi(B)\Phi(B^s) w_t = 	heta(B)\Theta(B^s) arepsilon_t\)
    private func computeMultiplicativePrediction(
        historyW: [Double],
        t: Int,
        historyE: [Double],
        p: Int,
        P: Int,
        q: Int,
        Q: Int,
        s: Int,
        ar: [Double],
        sar: [Double],
        ma: [Double],
        sma: [Double],
        intercept: Double
    ) -> Double {
        var val = intercept

        // 1. Regular AR terms: + phi_i * w[t - 1 - i]
        for i in 0..<p {
            let idx = t - 1 - i
            if idx >= 0 && idx < historyW.count {
                val += ar[i] * historyW[idx]
            }
        }

        // 2. Seasonal AR terms: + Phi_I * w[t - (I + 1) * s]
        for I in 0..<P {
            let idx = t - (I + 1) * s
            if idx >= 0 && idx < historyW.count {
                val += sar[I] * historyW[idx]
            }
        }

        // 3. Multiplicative AR cross-terms: - phi_i * Phi_I * w[t - 1 - i - (I + 1) * s]
        for i in 0..<p {
            for I in 0..<P {
                let idx = t - 1 - i - (I + 1) * s
                if idx >= 0 && idx < historyW.count {
                    val -= ar[i] * sar[I] * historyW[idx]
                }
            }
        }

        // 4. Regular MA terms: + theta_j * e[t - 1 - j]
        for j in 0..<q {
            let idx = t - 1 - j
            if idx >= 0 && idx < historyE.count {
                val += ma[j] * historyE[idx]
            }
        }

        // 5. Seasonal MA terms: + Theta_J * e[t - (J + 1) * s]
        for J in 0..<Q {
            let idx = t - (J + 1) * s
            if idx >= 0 && idx < historyE.count {
                val += sma[J] * historyE[idx]
            }
        }

        // 6. Multiplicative MA cross-terms: + theta_j * Theta_J * e[t - 1 - j - (J + 1) * s]
        for j in 0..<q {
            for J in 0..<Q {
                let idx = t - 1 - j - (J + 1) * s
                if idx >= 0 && idx < historyE.count {
                    val += ma[j] * sma[J] * historyE[idx]
                }
            }
        }

        return val
    }

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

        let solution = Array(b[0..<cols])
        try ForecastValidation.ensureFinite(solution)
        return solution
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
                // Closed-loop one-step reconstruction from observed subLevelSeries[i]
                reconstructed.reserveCapacity(subLevelSeries.count)
                reconstructed.append(subLevelSeries[0])
                for (i, diffVal) in current.enumerated() {
                    reconstructed.append(subLevelSeries[i] + diffVal)
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
                // Closed-loop seasonal one-step reconstruction from observed subLevelSeries[i]
                reconstructed.reserveCapacity(subLevelSeries.count)
                for i in 0..<Swift.min(s, subLevelSeries.count) {
                    reconstructed.append(subLevelSeries[i])
                }
                for (i, diffVal) in current.enumerated() {
                    reconstructed.append(subLevelSeries[i] + diffVal)
                }
            }
            current = reconstructed
        }

        return current
    }
}
