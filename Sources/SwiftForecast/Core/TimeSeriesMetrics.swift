import Foundation
import Accelerate

/// Canonical evaluation metrics for time-series forecasting and out-of-sample validation (G-020).
public enum TimeSeriesMetrics {

    /// Root Mean Squared Error (RMSE).
    /// - Parameters:
    ///   - actual: Ground-truth target values.
    ///   - forecast: Predicted forecast values.
    /// - Returns: RMSE scalar value.
    public static func rmse(actual: [Double], forecast: [Double]) -> Double {
        guard !actual.isEmpty, actual.count == forecast.count else { return 0.0 }
        var diff = [Double](repeating: 0.0, count: actual.count)
        vDSP.subtract(actual, forecast, result: &diff)
        let mse = vDSP.sumOfSquares(diff) / Double(actual.count)
        return sqrt(mse)
    }

    /// Mean Absolute Error (MAE).
    /// - Parameters:
    ///   - actual: Ground-truth target values.
    ///   - forecast: Predicted forecast values.
    /// - Returns: MAE scalar value.
    public static func mae(actual: [Double], forecast: [Double]) -> Double {
        guard !actual.isEmpty, actual.count == forecast.count else { return 0.0 }
        let sum = zip(actual, forecast).reduce(0.0) { $0 + abs($1.0 - $1.1) }
        return sum / Double(actual.count)
    }

    /// Mean Absolute Percentage Error (MAPE) expressed in percent.
    /// - Parameters:
    ///   - actual: Ground-truth target values.
    ///   - forecast: Predicted forecast values.
    ///   - epsilon: Threshold below which actuals are treated as zero to prevent division by zero.
    /// - Returns: MAPE percentage value [0, inf).
    public static func mape(actual: [Double], forecast: [Double], epsilon: Double = 1e-12) -> Double {
        guard !actual.isEmpty, actual.count == forecast.count else { return 0.0 }
        var sum = 0.0
        var count = 0
        for (y, yHat) in zip(actual, forecast) {
            let denom = abs(y)
            if denom > epsilon {
                sum += abs(y - yHat) / denom
                count += 1
            }
        }
        return count > 0 ? (sum / Double(count)) * 100.0 : 0.0
    }

    /// Symmetric Mean Absolute Percentage Error (SMAPE) expressed in percent [0, 200%].
    ///
    /// Evaluates relative forecast error bounded between 0% and 200%:
    /// $$\text{SMAPE} = \frac{100\%}{n} \sum_{t=1}^n \frac{2 |y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t| + \varepsilon}$$
    /// - Parameters:
    ///   - actual: Ground-truth target values.
    ///   - forecast: Predicted forecast values.
    ///   - epsilon: Small constant preventing division by zero when both values are zero.
    /// - Returns: SMAPE percentage value in [0, 200].
    public static func smape(actual: [Double], forecast: [Double], epsilon: Double = 1e-12) -> Double {
        guard !actual.isEmpty, actual.count == forecast.count else { return 0.0 }
        var sum = 0.0
        for (y, yHat) in zip(actual, forecast) {
            let denom = abs(y) + abs(yHat)
            if denom > epsilon {
                sum += (2.0 * abs(y - yHat)) / denom
            }
        }
        return (sum / Double(actual.count)) * 100.0
    }

    /// Computes the in-sample mean absolute error of a 1-step (or s-step seasonal) naive persistence forecast.
    ///
    /// This serves as the canonical scaling factor in the denominator of MASE (Hyndman & Koehler, 2006):
    /// $$\text{scale} = \frac{1}{T - s} \sum_{t=s+1}^T |y_t - y_{t-s}|$$
    /// - Parameters:
    ///   - trainingSeries: In-sample observations array.
    ///   - seasonality: Seasonal lag (default 1).
    /// - Returns: Mean absolute error of naive benchmark on training set.
    public static func naiveInSampleMAE(trainingSeries: [Double], seasonality: Int = 1) -> Double {
        let s = max(1, seasonality)
        let n = trainingSeries.count
        guard n > s else { return 0.0 }
        var naiveSum = 0.0
        for t in s..<n {
            naiveSum += abs(trainingSeries[t] - trainingSeries[t - s])
        }
        return naiveSum / Double(n - s)
    }

    /// Mean Absolute Scaled Error (MASE) scaled by in-sample 1-step naive seasonal baseline (Hyndman & Koehler, 2006).
    ///
    /// MASE < 1 indicates that the model outperforms the naive baseline on average.
    /// - Parameters:
    ///   - trainingSeries: In-sample training series used to compute scale factor.
    ///   - actual: Out-of-sample ground-truth target values.
    ///   - forecast: Out-of-sample predicted forecast values.
    ///   - seasonality: Seasonal lag for naive differencing (default is 1 for non-seasonal).
    /// - Returns: MASE scalar value.
    public static func mase(
        trainingSeries: [Double],
        actual: [Double],
        forecast: [Double],
        seasonality: Int = 1
    ) -> Double {
        let scale = naiveInSampleMAE(trainingSeries: trainingSeries, seasonality: seasonality)
        guard scale > 1e-12 else { return 0.0 }
        let maeVal = mae(actual: actual, forecast: forecast)
        return maeVal / scale
    }

    // MARK: - Validated Throwing Metrics

    /// Validates matching dimensions and non-empty inputs for time-series error metrics.
    private static func validateInputs(actual: [Double], forecast: [Double]) throws {
        guard !actual.isEmpty else { throw ForecastError.emptyTimeSeries }
        guard actual.count == forecast.count else {
            throw ForecastError.dimensionMismatch(expected: actual.count, got: forecast.count)
        }
    }

    /// Evaluates Root Mean Squared Error (RMSE), throwing on empty inputs or length mismatches.
    public static func evaluateRMSE(actual: [Double], forecast: [Double]) throws -> Double {
        try validateInputs(actual: actual, forecast: forecast)
        return rmse(actual: actual, forecast: forecast)
    }

    /// Evaluates Mean Absolute Error (MAE), throwing on empty inputs or length mismatches.
    public static func evaluateMAE(actual: [Double], forecast: [Double]) throws -> Double {
        try validateInputs(actual: actual, forecast: forecast)
        return mae(actual: actual, forecast: forecast)
    }

    /// Evaluates Mean Absolute Percentage Error (MAPE), throwing on empty inputs or length mismatches.
    public static func evaluateMAPE(actual: [Double], forecast: [Double], epsilon: Double = 1e-12) throws -> Double {
        try validateInputs(actual: actual, forecast: forecast)
        return mape(actual: actual, forecast: forecast, epsilon: epsilon)
    }

    /// Evaluates Symmetric Mean Absolute Percentage Error (SMAPE), throwing on empty inputs or length mismatches.
    public static func evaluateSMAPE(actual: [Double], forecast: [Double], epsilon: Double = 1e-12) throws -> Double {
        try validateInputs(actual: actual, forecast: forecast)
        return smape(actual: actual, forecast: forecast, epsilon: epsilon)
    }

    /// Evaluates Mean Absolute Scaled Error (MASE), throwing on empty inputs or length mismatches.
    public static func evaluateMASE(
        trainingSeries: [Double],
        actual: [Double],
        forecast: [Double],
        seasonality: Int = 1
    ) throws -> Double {
        guard !trainingSeries.isEmpty else { throw ForecastError.emptyTimeSeries }
        try validateInputs(actual: actual, forecast: forecast)
        let scale = naiveInSampleMAE(trainingSeries: trainingSeries, seasonality: seasonality)
        guard scale > 1e-12 else {
            throw ForecastError.invalidParameter("Naive in-sample MAE scale is zero (constant training series); MASE is undefined.")
        }
        return try evaluateMAE(actual: actual, forecast: forecast) / scale
    }
}
