import Testing
import Foundation
@testable import SwiftForecast

@Suite("Rolling-Origin Time Series Backtester Tests (G-020)")
struct BacktestTests {

    @Test("TimeSeriesMetrics calculates canonical RMSE, MAE, MAPE, SMAPE, MASE")
    func testMetricsFormulas() {
        let actual = [10.0, 20.0, 30.0]
        let forecast = [12.0, 18.0, 33.0]

        let rmse = TimeSeriesMetrics.rmse(actual: actual, forecast: forecast)
        let mae = TimeSeriesMetrics.mae(actual: actual, forecast: forecast)
        let mape = TimeSeriesMetrics.mape(actual: actual, forecast: forecast)
        let smape = TimeSeriesMetrics.smape(actual: actual, forecast: forecast)

        #expect(abs(mae - (2.0 + 2.0 + 3.0) / 3.0) < 1e-6)
        #expect(abs(rmse - sqrt((4.0 + 4.0 + 9.0) / 3.0)) < 1e-6)
        #expect(mape > 0.0)
        #expect(smape > 0.0 && smape < 100.0)

        let train = [2.0, 4.0, 6.0, 8.0, 10.0]
        let mase = TimeSeriesMetrics.mase(trainingSeries: train, actual: actual, forecast: forecast)
        #expect(mase > 0.0)
    }

    @Test("Rolling-origin backtesting with Moving Average baseline predictor")
    func testMovingAverageBacktest() async throws {
        let series = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        let result = try await TimeSeriesBacktester.backtestMovingAverage(
            series: series,
            window: 3,
            initialWindow: 5,
            horizon: 1,
            stepSize: 1,
            mode: .expanding
        )

        #expect(result.folds.count == 5)
        #expect(result.rmse > 0.0)
        #expect(result.mae > 0.0)
        #expect(result.mape > 0.0)
        #expect(result.smape > 0.0)
        #expect(result.mase > 0.0)
        #expect(result.metrics["rmse"] == result.rmse)
    }

    @Test("Rolling-origin backtesting with Exponential Smoothing")
    func testExponentialSmoothingBacktest() async throws {
        let series: [Double] = (0..<20).map { Double($0) * 1.5 + Double($0 % 3) }
        let result = try await TimeSeriesBacktester.backtestExponentialSmoothing(
            series: series,
            method: .double(beta: 0.1),
            initialWindow: 10,
            horizon: 2,
            stepSize: 2,
            mode: .sliding(windowSize: 10)
        )

        #expect(result.folds.count == 5)
        #expect(result.rmse > 0.0)
        #expect(result.mae > 0.0)
    }

    @Test("Rolling-origin backtesting with ARIMA")
    func testARIMABacktest() async throws {
        let series: [Double] = (0..<25).map { 100.0 + Double($0) * 0.5 }
        let result = try await TimeSeriesBacktester.backtestARIMA(
            series: series,
            p: 1, d: 1, q: 0,
            initialWindow: 15,
            horizon: 1,
            stepSize: 2,
            mode: .expanding
        )

        #expect(result.folds.count == 5)
        #expect(result.rmse >= 0.0)
        #expect(result.smape >= 0.0)
    }
}
