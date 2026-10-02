import Foundation

/// Backtesting window evaluation mode along the temporal axis.
public enum BacktestWindowMode: Sendable, Codable, Equatable {
    /// Expanding origin window: each fold uses all historical data from index 0 to current origin.
    case expanding
    /// Sliding fixed-size window: each fold uses the most recent `windowSize` observations up to the origin.
    case sliding(windowSize: Int)
}

/// A single evaluation fold in rolling-origin backtesting.
public struct BacktestFold: Sendable, Codable, Equatable {
    /// The origin time index (number of observations in training series up to origin).
    public let originIndex: Int
    /// Forecast horizon for this fold.
    public let horizon: Int
    /// Actual out-of-sample ground truth values for the horizon.
    public let actual: [Double]
    /// Predicted out-of-sample values for the horizon.
    public let forecast: [Double]
    /// Fold Root Mean Squared Error.
    public let rmse: Double
    /// Fold Mean Absolute Error.
    public let mae: Double
    /// Fold Mean Absolute Percentage Error.
    public let mape: Double
    /// Fold Symmetric Mean Absolute Percentage Error.
    public let smape: Double
}

/// Comprehensive rolling-origin backtest evaluation results (G-020).
public struct TimeSeriesBacktestResult: Sendable, Codable, Equatable {
    /// Per-fold evaluations across all rolling origins.
    public let folds: [BacktestFold]
    /// Aggregate out-of-sample RMSE across all evaluated prediction points.
    public let rmse: Double
    /// Aggregate out-of-sample MAE across all evaluated prediction points.
    public let mae: Double
    /// Aggregate out-of-sample MAPE in percent.
    public let mape: Double
    /// Aggregate out-of-sample SMAPE in percent.
    public let smape: Double
    /// Aggregate out-of-sample MASE against in-sample naive baseline.
    public let mase: Double
    /// Flat dictionary of summary evaluation metrics for reporting.
    public let metrics: [String: Double]

    /// Creates a new TimeSeriesBacktestResult instance.
    public init(
        folds: [BacktestFold],
        rmse: Double,
        mae: Double,
        mape: Double,
        smape: Double,
        mase: Double,
        metrics: [String: Double]? = nil
    ) {
        self.folds = folds
        self.rmse = rmse
        self.mae = mae
        self.mape = mape
        self.smape = smape
        self.mase = mase
        if let m = metrics {
            self.metrics = m
        } else {
            self.metrics = [
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "smape": smape,
                "mase": mase
            ]
        }
    }
}

/// Standardized rolling-origin (expanding / sliding window) backtesting utility (G-020).
///
/// Evaluates out-of-sample forecasting performance across moving origins without requiring
/// consumers to write manual partitioning and horizon iteration loops.
public enum TimeSeriesBacktester {

    /// Performs rolling-origin backtesting using a custom model forecast closure.
    /// - Parameters:
    ///   - series: Full historical time series observations array.
    ///   - initialWindow: Number of observations in the initial training split.
    ///   - horizon: Forecast step count evaluated at each origin (default 1).
    ///   - stepSize: Step stride between consecutive origins (default 1).
    ///   - mode: `.expanding` or `.sliding(windowSize:)` window configuration.
    ///   - fitAndForecast: Asynchronous closure taking `(trainingSeries, horizon)` and returning predicted values.
    /// - Returns: `TimeSeriesBacktestResult` with per-fold metrics and pooled out-of-sample error statistics.
    /// - Throws: `ForecastError` if series is too short or invalid parameters are provided.
    public static func backtest(
        series: [Double],
        initialWindow: Int,
        horizon: Int = 1,
        stepSize: Int = 1,
        mode: BacktestWindowMode = .expanding,
        fitAndForecast: @Sendable (_ trainSeries: [Double], _ horizon: Int) async throws -> [Double]
    ) async throws -> TimeSeriesBacktestResult {
        guard !series.isEmpty else { throw ForecastError.emptyTimeSeries }
        guard initialWindow >= 2 else {
            throw ForecastError.insufficientLength(minimum: 2, got: initialWindow)
        }
        guard horizon >= 1 else {
            throw ForecastError.invalidHorizon(horizon)
        }
        guard stepSize >= 1 else {
            throw ForecastError.invalidParameter("stepSize must be >= 1, got \(stepSize)")
        }
        let total = series.count
        guard total >= initialWindow + horizon else {
            throw ForecastError.insufficientLength(minimum: initialWindow + horizon, got: total)
        }

        var folds: [BacktestFold] = []
        var allActuals: [Double] = []
        var allForecasts: [Double] = []

        var origin = initialWindow
        while origin + horizon <= total {
            let trainSeries: [Double]
            switch mode {
            case .expanding:
                trainSeries = Array(series[0..<origin])
            case .sliding(let windowSize):
                let actualWindow = min(windowSize, origin)
                trainSeries = Array(series[(origin - actualWindow)..<origin])
            }

            let actual = Array(series[origin..<(origin + horizon)])
            let predicted = try await fitAndForecast(trainSeries, horizon)

            let foldRmse = TimeSeriesMetrics.rmse(actual: actual, forecast: predicted)
            let foldMae = TimeSeriesMetrics.mae(actual: actual, forecast: predicted)
            let foldMape = TimeSeriesMetrics.mape(actual: actual, forecast: predicted)
            let foldSmape = TimeSeriesMetrics.smape(actual: actual, forecast: predicted)

            let fold = BacktestFold(
                originIndex: origin,
                horizon: horizon,
                actual: actual,
                forecast: predicted,
                rmse: foldRmse,
                mae: foldMae,
                mape: foldMape,
                smape: foldSmape
            )
            folds.append(fold)
            allActuals.append(contentsOf: actual)
            allForecasts.append(contentsOf: predicted)

            origin += stepSize
        }

        let pooledRmse = TimeSeriesMetrics.rmse(actual: allActuals, forecast: allForecasts)
        let pooledMae = TimeSeriesMetrics.mae(actual: allActuals, forecast: allForecasts)
        let pooledMape = TimeSeriesMetrics.mape(actual: allActuals, forecast: allForecasts)
        let pooledSmape = TimeSeriesMetrics.smape(actual: allActuals, forecast: allForecasts)
        let initialTrain = Array(series[0..<initialWindow])
        let pooledMase = TimeSeriesMetrics.mase(
            trainingSeries: initialTrain,
            actual: allActuals,
            forecast: allForecasts
        )

        return TimeSeriesBacktestResult(
            folds: folds,
            rmse: pooledRmse,
            mae: pooledMae,
            mape: pooledMape,
            smape: pooledSmape,
            mase: pooledMase
        )
    }

    /// Performs rolling-origin backtesting with `ARIMAModel`.
    public static func backtestARIMA(
        series: [Double],
        p: Int, d: Int, q: Int,
        initialWindow: Int,
        horizon: Int = 1,
        stepSize: Int = 1,
        mode: BacktestWindowMode = .expanding
    ) async throws -> TimeSeriesBacktestResult {
        try await backtest(
            series: series,
            initialWindow: initialWindow,
            horizon: horizon,
            stepSize: stepSize,
            mode: mode
        ) { train, h in
            let model = try ARIMAModel(p: p, d: d, q: q)
            try await model.fit(series: train)
            let res = try await model.forecast(horizon: h)
            return res.forecast.predictions
        }
    }

    /// Performs rolling-origin backtesting with `ExponentialSmoothing`.
    public static func backtestExponentialSmoothing(
        series: [Double],
        method: SmoothingMethod,
        initialWindow: Int,
        horizon: Int = 1,
        stepSize: Int = 1,
        mode: BacktestWindowMode = .expanding
    ) async throws -> TimeSeriesBacktestResult {
        try await backtest(
            series: series,
            initialWindow: initialWindow,
            horizon: horizon,
            stepSize: stepSize,
            mode: mode
        ) { train, h in
            let model = ExponentialSmoothing(method: method)
            try await model.fit(series: train)
            let res = try await model.forecast(horizon: h)
            return res.predictions
        }
    }

    /// Performs rolling-origin backtesting with a standard Moving Average baseline predictor.
    public static func backtestMovingAverage(
        series: [Double],
        window: Int = 3,
        initialWindow: Int,
        horizon: Int = 1,
        stepSize: Int = 1,
        mode: BacktestWindowMode = .expanding
    ) async throws -> TimeSeriesBacktestResult {
        let w = max(1, window)
        return try await backtest(
            series: series,
            initialWindow: initialWindow,
            horizon: horizon,
            stepSize: stepSize,
            mode: mode
        ) { train, h in
            let slice = train.suffix(w)
            let meanVal = slice.reduce(0.0, +) / Double(slice.count)
            return [Double](repeating: meanVal, count: h)
        }
    }
}
