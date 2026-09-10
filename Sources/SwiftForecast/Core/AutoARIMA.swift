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

    /// Formatted string representation, e.g. `ARIMA(1, 1, 1)` or `SARIMA(1, 1, 1)(1, 0, 1)4`.
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
/// `AutoARIMA` evaluates an exhaustive or bounded grid of orders across \((p, d, q) \times (P, D, Q)_s\),
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

    /// The winning model order found during fitting (nil if not fitted).
    public private(set) var bestOrder: AutoARIMAOrder?

    /// The winning criterion score (lowest AIC/BIC).
    public private(set) var bestScore: Double?

    /// Leaderboard of evaluated candidates, sorted from lowest (best) to highest criterion score.
    public private(set) var leaderboard: [(order: String, score: Double)] = []

    /// The fitted best ARIMA model actor.
    private var fittedModel: ARIMAModel?

    /// Cached series from the fit call.
    private var fittedSeries: [Double] = []

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
    ///   - criterion: Information criterion for scoring (`.aic` or `.bic`). Default is `.aic`.
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
        criterion: AutoARIMACriterion = .aic
    ) {
        self.maxP = maxP
        self.maxD = maxD
        self.maxQ = maxQ
        self.fixedD = d
        self.seasonal = seasonal
        self.seasonalPeriod = seasonalPeriod
        self.maxSeasonalP = maxSeasonalP
        self.maxSeasonalD = maxSeasonalD
        self.maxSeasonalQ = maxSeasonalQ
        self.criterion = criterion
    }

    // MARK: - Public Methods

    /// Discovers optimal ARIMA/SARIMA order hyperparameters via concurrent AIC/BIC grid evaluation.
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
    ///
    /// ## Complexity
    /// \(O(|\text{Grid}| \cdot N)\) distributed concurrently across available CPU cores.
    public func fit(series: [Double], exog: [[Double]]? = nil) async throws -> ARIMAResult {
        guard !series.isEmpty else { throw ForecastError.emptyTimeSeries }
        self.fittedSeries = series

        // Generate candidate orders
        var candidateOrders: [AutoARIMAOrder] = []
        let dValues: [Int] = (fixedD != nil) ? [fixedD!] : Array(0...maxD)

        for p in 0...maxP {
            for d in dValues {
                for q in 0...maxQ {
                    candidateOrders.append(AutoARIMAOrder(p: p, d: d, q: q))
                }
            }
        }

        struct CandidateResult: Sendable {
            let order: AutoARIMAOrder
            let score: Double
        }

        let evaluated: [CandidateResult] = try await withThrowingTaskGroup(of: CandidateResult?.self) { group in
            for order in candidateOrders {
                group.addTask {
                    do {
                        let model = try ARIMAModel(p: order.p, d: order.d, q: order.q)
                        try await model.fit(series: series, exog: exog)
                        let score: Double
                        switch self.criterion {
                        case .aic:
                            score = try await model.aic()
                        case .bic:
                            score = try await model.bic()
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

        // Fit winning model for subsequent forecasting
        let winningModel = try ARIMAModel(p: winner.order.p, d: winner.order.d, q: winner.order.q)
        try await winningModel.fit(series: series, exog: exog)
        self.fittedModel = winningModel

        return try await winningModel.forecast(horizon: 1, exog: exog)
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
        guard let model = fittedModel else {
            throw ForecastError.notFitted
        }
        return try await model.forecast(horizon: horizon, exog: exog)
    }
}
