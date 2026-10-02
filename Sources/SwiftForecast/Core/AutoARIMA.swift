import Foundation
import Accelerate

/// Information criterion utilized for model selection in `AutoARIMA`.
public enum AutoARIMACriterion: String, Sendable, Codable {
    /// Akaike Information Criterion: \(2k - 2\ln(L)\).
    case aic
    /// Bayesian Information Criterion: \(k\ln(N) - 2\ln(L)\).
    case bic
}

/// A structure identifying the selected ARIMA / SARIMA order.
public struct AutoARIMAOrder: Sendable, Equatable, CustomStringConvertible {
    /// Non-seasonal autoregressive order \(p\).
    public let p: Int
    /// Non-seasonal degree of differencing \(d\).
    public let d: Int
    /// Non-seasonal moving average order \(q\).
    public let q: Int
    /// Seasonal autoregressive order \(P\) (0 if non-seasonal).
    public let P: Int
    /// Seasonal degree of differencing \(D\) (0 if non-seasonal).
    public let D: Int
    /// Seasonal moving average order \(Q\) (0 if non-seasonal).
    public let Q: Int
    /// Seasonal period \(s\) (1 if non-seasonal).
    public let s: Int

    /// Formatted string representation, e.g. `ARIMA(1, 1, 1)` or `SARIMA(1, 1, 1)(1, 0, 1)[4]`.
    public var description: String {
        if s > 1 || P > 0 || D > 0 || Q > 0 {
            return "SARIMA(\(p), \(d), \(q))(\(P), \(D), \(Q))[\(s)]"
        } else {
            return "ARIMA(\(p), \(d), \(q))"
        }
    }

    /// Creates a new order specification.
    ///
    /// - Parameters:
    ///   - p: Autoregressive order.
    ///   - d: Differencing order.
    ///   - q: Moving average order.
    ///   - P: Seasonal autoregressive order. Default is 0.
    ///   - D: Seasonal differencing order. Default is 0.
    ///   - Q: Seasonal moving average order. Default is 0.
    ///   - s: Seasonal period. Default is 1.
    public init(p: Int, d: Int, q: Int, P: Int = 0, D: Int = 0, Q: Int = 0, s: Int = 1) {
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
    }
}

/// Automated hyperparameter grid search for ARIMA and SARIMA time-series models.
///
/// `AutoARIMA` evaluates an exhaustive or bounded grid of orders across \((p, d, q) 	imes (P, D, Q)_s\),
/// identifying the optimal order that minimizes AIC or BIC while ensuring stability and invertibility.
///
/// ## Concurrency Management
/// Evaluates candidate model configurations concurrently across Apple Silicon CPU cores via structured `withThrowingTaskGroup`.
///
/// ## Thread Safety
/// Implemented as an isolated Swift actor ensuring thread safety under Swift 6 strict concurrency.
public actor AutoARIMA {

    /// Maximum non-seasonal autoregressive order \(p\).
    public let maxP: Int

    /// Maximum non-seasonal differencing order \(d\).
    public let maxD: Int

    /// Maximum non-seasonal moving average order \(q\).
    public let maxQ: Int

    /// Fixed differencing order, if specified.
    public let fixedD: Int?

    /// Enables seasonal SARIMA grid search when true.
    public let seasonal: Bool

    /// Seasonal period length \(s\) (e.g. 4 for quarterly, 12 for monthly, 7 for weekly).
    public let seasonalPeriod: Int

    /// Maximum seasonal autoregressive order \(P\).
    public let maxSeasonalP: Int

    /// Maximum seasonal differencing order \(D\).
    public let maxSeasonalD: Int

    /// Maximum seasonal moving average order \(Q\).
    public let maxSeasonalQ: Int

    /// Information criterion used to score and rank models.
    public let criterion: AutoARIMACriterion

    /// Maximum number of candidate models evaluated in order search (prevents infinite loops and task starvation).
    ///
    /// > Important: Candidate orders are sorted by parsimonious model complexity (\(p+d+q+P+D+Q\)),
    /// > so `maxCandidates` heuristically truncates the search space to the \(N\) simplest candidate models
    /// > rather than performing a random subsampling or exhaustive evaluation of the entire grid.
    public let maxCandidates: Int

    /// Deprecated backwards-compatible alias for `maxCandidates`.
    @available(*, deprecated, renamed: "maxCandidates")
    public var maxIterations: Int {
        maxCandidates
    }

    /// The winning model order found during fitting (nil if not fitted).
    public private(set) var bestOrder: AutoARIMAOrder?

    /// The winning criterion score (lowest AIC/BIC).
    public private(set) var bestScore: Double?

    /// Leaderboard of evaluated candidates, sorted from lowest (best) to highest criterion score.
    public private(set) var leaderboard: [(order: String, score: Double)] = []

    /// The fitted best ARIMA model actor.
    private var fittedModel: ARIMAModel?

    /// The fitted best SARIMA model actor.
    private var fittedSARIMAModel: SARIMAModel?

    /// Cached series from the fit call.
    private var fittedSeries: [Double]?

    // MARK: - Initialization

    /// Creates a new `AutoARIMA` automated model selector.
    ///
    /// - Parameters:
    ///   - maxP: Maximum autoregressive order \(p\). Default is `3`.
    ///   - maxD: Maximum differencing degree \(d\). Default is `2`.
    ///   - maxQ: Maximum moving average order \(q\). Default is `3`.
    ///   - d: Optional fixed differencing degree \(d\). If specified, overrides `maxD`. Default is `nil`.
    ///   - seasonal: Whether to search seasonal SARIMA orders. Default is `false`.
    ///   - seasonalPeriod: Seasonal period \(s\). Default is `1`.
    ///   - maxSeasonalP: Maximum seasonal autoregressive order \(P\). Default is `1`.
    ///   - maxSeasonalD: Maximum seasonal differencing order \(D\). Default is `1`.
    ///   - maxSeasonalQ: Maximum seasonal moving average order \(Q\). Default is `1`.
    ///   - maxCandidates: Maximum number of candidate models evaluated. The candidate grid is
    ///     ordered by parsimony (\(p+d+q+P+D+Q\)), and this parameter truncates evaluation to the
    ///     first `maxCandidates` simplest models (heuristic parsimonious truncation). Default is `100`.
    ///   - criterion: Information criterion for scoring (`.aic` or `.bic`). Default is `.aic`.
    /// - Throws: `ForecastError` if configuration bounds or orders are invalid.
    public init(
        maxP: Int = 3,
        maxD: Int = 2,
        maxQ: Int = 3,
        d: Int? = nil,
        seasonal: Bool = false,
        seasonalPeriod: Int = 1,
        maxSeasonalP: Int = 1,
        maxSeasonalD: Int = 1,
        maxSeasonalQ: Int = 1,
        maxCandidates: Int = 100,
        criterion: AutoARIMACriterion = .aic
    ) throws {
        guard maxP >= 0 else { throw ForecastError.invalidAROrder(maxP) }
        guard maxD >= 0 else { throw ForecastError.invalidDifferencing(maxD) }
        guard maxQ >= 0 else { throw ForecastError.invalidMAOrder(maxQ) }
        if let fixedD = d {
            guard fixedD >= 0 else { throw ForecastError.invalidDifferencing(fixedD) }
        }
        if seasonal {
            guard seasonalPeriod >= 2 else { throw ForecastError.invalidSeasonalPeriod(seasonalPeriod) }
            guard maxSeasonalP >= 0 else { throw ForecastError.invalidAROrder(maxSeasonalP) }
            guard maxSeasonalD >= 0 else { throw ForecastError.invalidDifferencing(maxSeasonalD) }
            guard maxSeasonalQ >= 0 else { throw ForecastError.invalidMAOrder(maxSeasonalQ) }
        }
        guard maxCandidates > 0 else {
            throw ForecastError.insufficientLength(minimum: 1, got: maxCandidates)
        }

        self.maxP = maxP
        self.maxD = maxD
        self.maxQ = maxQ
        self.fixedD = d
        self.seasonal = seasonal
        self.seasonalPeriod = seasonalPeriod
        self.maxSeasonalP = maxSeasonalP
        self.maxSeasonalD = maxSeasonalD
        self.maxSeasonalQ = maxSeasonalQ
        self.maxCandidates = maxCandidates
        self.criterion = criterion
    }

    /// Backwards-compatible convenience initializer using `maxIterations`.
    @available(*, deprecated, renamed: "init(maxP:maxD:maxQ:d:seasonal:seasonalPeriod:maxSeasonalP:maxSeasonalD:maxSeasonalQ:maxCandidates:criterion:)")
    public init(
        maxP: Int = 3,
        maxD: Int = 2,
        maxQ: Int = 3,
        d: Int? = nil,
        seasonal: Bool = false,
        seasonalPeriod: Int = 1,
        maxSeasonalP: Int = 1,
        maxSeasonalD: Int = 1,
        maxSeasonalQ: Int = 1,
        maxIterations: Int,
        criterion: AutoARIMACriterion = .aic
    ) throws {
        try self.init(
            maxP: maxP,
            maxD: maxD,
            maxQ: maxQ,
            d: d,
            seasonal: seasonal,
            seasonalPeriod: seasonalPeriod,
            maxSeasonalP: maxSeasonalP,
            maxSeasonalD: maxSeasonalD,
            maxSeasonalQ: maxSeasonalQ,
            maxCandidates: maxIterations,
            criterion: criterion
        )
    }

    // MARK: - Stationarity & Invertibility Checks

    /// Verifies whether an AR or MA polynomial is stable/invertible (roots strictly outside the unit circle)
    /// using the Schur-Cohn / Levinson-Durbin step-down reflection coefficient recursion.
    public static func isPolynomialStable(coefficients: [Double]) -> Bool {
        guard !coefficients.isEmpty else { return true }
        var a = coefficients
        var p = a.count
        while p > 0 {
            let r = a[p - 1]
            if abs(r) >= 1.0 - 1e-7 {
                return false
            }
            let denom = 1.0 - r * r
            if denom <= 1e-15 {
                return false
            }
            var nextA = [Double](repeating: 0.0, count: p - 1)
            for j in 0..<(p - 1) {
                nextA[j] = (a[j] + r * a[p - 2 - j]) / denom
            }
            a = nextA
            p -= 1
        }
        return true
    }

    // MARK: - Public Methods

    /// Discovers optimal ARIMA/SARIMA order hyperparameters via concurrent AIC/BIC grid evaluation.
    ///
    /// > Note: Candidate orders are sorted by parsimonious complexity (\(p+d+q+P+D+Q\)).
    /// > If `maxCandidates` is set below the total grid size, search evaluates the simplest `maxCandidates`
    /// > configurations (heuristic parsimonious truncation).
    ///
    /// ## Concurrency Management
    /// Evaluates order candidates concurrently across thread pools via structured `withThrowingTaskGroup`.
    ///
    /// ## Thread Safety
    /// Thread-safe via actor isolation.
    ///
    /// - Parameters:
    ///   - series: Array of historical time series observations.
    ///   - exog: Optional 2D array of exogenous regressor features.
    /// - Returns: An `ARIMAResult` containing the winning model's in-sample fit and diagnostics.
    /// - Throws: `ForecastError` if series is too short or all candidate configurations fail to converge.
    public func fit(series: [Double], exog: [[Double]]? = nil) async throws -> ARIMAResult {
        guard !series.isEmpty else { throw ForecastError.emptyTimeSeries }
        if seasonal && seasonalPeriod > 1 && exog != nil && !exog!.isEmpty {
            throw ForecastError.invalidParameter("Seasonal AutoARIMA with exogenous regressors (SARIMAX) is currently unsupported. Omit 'exog' or set 'seasonal: false' for ARIMAX.")
        }
        self.fittedSeries = series

        // Zero-variance guard: if the series is constant, fit trivial ARIMA(0, 0, 0)
        let minVal = series.min() ?? 0.0
        let maxVal = series.max() ?? 0.0
        if abs(maxVal - minVal) < 1e-12 {
            let zeroOrder = AutoARIMAOrder(p: 0, d: 0, q: 0)
            self.bestOrder = zeroOrder
            self.bestScore = 0.0
            self.leaderboard = [(zeroOrder.description, 0.0)]
            let model = try ARIMAModel(p: 0, d: 0, q: 0)
            try await model.fit(series: series, exog: exog)
            self.fittedModel = model
            self.fittedSARIMAModel = nil
            let forecastExog = exog.flatMap { matrix in matrix.last.map { [$0] } }
            return try await model.forecast(horizon: 1, exog: forecastExog)
        }

        // Generate candidate orders
        var candidateOrders: [AutoARIMAOrder] = []
        let dValues: [Int] = (fixedD != nil) ? [fixedD!] : Array(0...maxD)

        if seasonal && seasonalPeriod > 1 {
            let sDValues = Array(0...maxSeasonalD)
            for p in 0...maxP {
                for d in dValues {
                    for q in 0...maxQ {
                        for P in 0...maxSeasonalP {
                            for D in sDValues {
                                for Q in 0...maxSeasonalQ {
                                    candidateOrders.append(
                                        AutoARIMAOrder(p: p, d: d, q: q, P: P, D: D, Q: Q, s: seasonalPeriod)
                                    )
                                }
                            }
                        }
                    }
                }
            }
        } else {
            for p in 0...maxP {
                for d in dValues {
                    for q in 0...maxQ {
                        candidateOrders.append(AutoARIMAOrder(p: p, d: d, q: q))
                    }
                }
            }
        }

        // Sort candidates by parsimony: simpler models evaluated before complex ones
        candidateOrders.sort {
            let sum0 = $0.p + $0.d + $0.q + $0.P + $0.D + $0.Q
            let sum1 = $1.p + $1.d + $1.q + $1.P + $1.D + $1.Q
            if sum0 != sum1 { return sum0 < sum1 }
            if $0.d != $1.d { return $0.d < $1.d }
            if $0.D != $1.D { return $0.D < $1.D }
            if $0.p != $1.p { return $0.p < $1.p }
            return $0.q < $1.q
        }

        let boundedCandidates = Array(candidateOrders.prefix(maxCandidates))

        struct CandidateResult: Sendable {
            let order: AutoARIMAOrder
            let score: Double
        }

        let evaluated: [CandidateResult] = try await withThrowingTaskGroup(of: CandidateResult?.self) { group in
            for order in boundedCandidates {
                group.addTask {
                    do {
                        let score: Double
                        if order.s > 1 && (order.P > 0 || order.D > 0 || order.Q > 0) {
                            let model = try SARIMAModel(
                                p: order.p,
                                d: order.d,
                                q: order.q,
                                P: order.P,
                                D: order.D,
                                Q: order.Q,
                                s: order.s
                            )
                            try await model.fit(series: series)

                            let arCoeffs = await model.arCoefficients
                            let maCoeffs = await model.maCoefficients
                            let sArCoeffs = await model.seasonalArCoefficients
                            let sMaCoeffs = await model.seasonalMaCoefficients

                            guard AutoARIMA.isPolynomialStable(coefficients: arCoeffs),
                                  AutoARIMA.isPolynomialStable(coefficients: maCoeffs),
                                  AutoARIMA.isPolynomialStable(coefficients: sArCoeffs),
                                  AutoARIMA.isPolynomialStable(coefficients: sMaCoeffs) else {
                                return nil
                            }

                            switch self.criterion {
                            case .aic:
                                score = try await model.aic()
                            case .bic:
                                score = try await model.bic()
                            }
                        } else {
                            let model = try ARIMAModel(p: order.p, d: order.d, q: order.q)
                            try await model.fit(series: series, exog: exog)

                            let arCoeffs = await model.arCoefficients
                            let maCoeffs = await model.maCoefficients

                            guard AutoARIMA.isPolynomialStable(coefficients: arCoeffs),
                                  AutoARIMA.isPolynomialStable(coefficients: maCoeffs) else {
                                return nil
                            }

                            switch self.criterion {
                            case .aic:
                                score = try await model.aic()
                            case .bic:
                                score = try await model.bic()
                            }
                        }

                        if score.isFinite && !score.isNaN {
                            return CandidateResult(order: order, score: score)
                        } else {
                            return nil
                        }
                    } catch {
                        // Inadmissible or non-convergent order; discard safely
                        return nil
                    }
                }
            }

            var results: [CandidateResult] = []
            for try await res in group {
                if let r = res {
                    results.append(r)
                }
            }
            return results
        }

        guard !evaluated.isEmpty else {
            throw ForecastError.trainingFailed("All AutoARIMA candidate models failed to converge on the series")
        }

        // Sort: lower AIC/BIC is superior
        let sorted = evaluated.sorted { $0.score < $1.score }
        self.leaderboard = sorted.map { ($0.order.description, $0.score) }

        let winner = sorted.first!
        self.bestOrder = winner.order
        self.bestScore = winner.score

        let isSeasonalWinner = winner.order.s > 1 && (winner.order.P > 0 || winner.order.D > 0 || winner.order.Q > 0)
        if isSeasonalWinner {
            let winningModel = try SARIMAModel(
                p: winner.order.p,
                d: winner.order.d,
                q: winner.order.q,
                P: winner.order.P,
                D: winner.order.D,
                Q: winner.order.Q,
                s: winner.order.s
            )
            try await winningModel.fit(series: series)
            self.fittedSARIMAModel = winningModel
            self.fittedModel = nil

            let preds = try await winningModel.forecast(steps: 1)
            let aicVal = (try? await winningModel.aic()) ?? 0.0
            let res = await winningModel.residuals
            let fittedVals = await winningModel.fittedValues
            let validRes = res.filter { $0.isFinite }
            let mseVal = validRes.isEmpty ? 0.0 : (vDSP.sumOfSquares(validRes) / Double(validRes.count))
            let maeVal = validRes.isEmpty ? 0.0 : (validRes.reduce(0.0) { $0 + abs($1) } / Double(validRes.count))

            let forecastResult = ForecastResult(
                predictions: preds,
                lowerBound: nil,
                upperBound: nil,
                fittedValues: fittedVals,
                residuals: res,
                aic: aicVal,
                mse: mseVal,
                mae: maeVal
            )
            return ARIMAResult(
                order: (winner.order.p, winner.order.d, winner.order.q),
                seasonalOrder: (winner.order.P, winner.order.D, winner.order.Q, winner.order.s),
                arCoefficients: await winningModel.arCoefficients,
                maCoefficients: await winningModel.maCoefficients,
                intercept: await winningModel.intercept,
                exogCoefficients: [],
                forecast: forecastResult
            )
        } else {
            let winningModel = try ARIMAModel(p: winner.order.p, d: winner.order.d, q: winner.order.q)
            try await winningModel.fit(series: series, exog: exog)
            self.fittedModel = winningModel
            self.fittedSARIMAModel = nil

            let forecastExog = exog.flatMap { matrix in matrix.last.map { [$0] } }
            return try await winningModel.forecast(horizon: 1, exog: forecastExog)
        }
    }

    /// Generates out-of-sample forecasts from the winning fitted model.
    ///
    /// - Parameters:
    ///   - horizon: Number of future time steps to forecast.
    ///   - exog: Optional exogenous feature matrix for future horizon steps.
    /// - Returns: An `ARIMAResult` with future predictions and fitted metrics.
    /// - Throws: `ForecastError.notFitted` if `fit()` has not been invoked.
    ///
    /// ## Thread Safety
    /// Thread-safe via actor isolation.
    public func forecast(horizon: Int, exog: [[Double]]? = nil) async throws -> ARIMAResult {
        if let sarima = fittedSARIMAModel {
            guard horizon >= 1 else { throw ForecastError.invalidHorizon(horizon) }
            let preds = try await sarima.forecast(steps: horizon)
            let order = sarima.order
            let sOrder = sarima.seasonalOrder
            let aicVal = (try? await sarima.aic()) ?? 0.0
            let res = await sarima.residuals
            let fittedVals = await sarima.fittedValues
            let validRes = res.filter { $0.isFinite }
            let mseVal = validRes.isEmpty ? 0.0 : (vDSP.sumOfSquares(validRes) / Double(validRes.count))
            let maeVal = validRes.isEmpty ? 0.0 : (validRes.reduce(0.0) { $0 + abs($1) } / Double(validRes.count))

            let forecastResult = ForecastResult(
                predictions: preds,
                lowerBound: nil,
                upperBound: nil,
                fittedValues: fittedVals,
                residuals: res,
                aic: aicVal,
                mse: mseVal,
                mae: maeVal
            )
            return ARIMAResult(
                order: order,
                seasonalOrder: sOrder,
                arCoefficients: await sarima.arCoefficients,
                maCoefficients: await sarima.maCoefficients,
                intercept: await sarima.intercept,
                exogCoefficients: [],
                forecast: forecastResult
            )
        } else if let model = fittedModel {
            return try await model.forecast(horizon: horizon, exog: exog)
        } else {
            throw ForecastError.notFitted
        }
    }
}
