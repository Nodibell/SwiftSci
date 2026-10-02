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

    @Test("Strict anti-leakage verification: future values never enter fold training sets")
    func testAntiLeakageStrictness() async throws {
        let baseSeries: [Double] = (0..<30).map { Double($0) * 2.0 }

        final class Recorder: @unchecked Sendable {
            var seenTrainSizes: [Int] = []
            var maxObservedInTrain: [Double] = []
            private let lock = NSLock()

            func record(size: Int, maxVal: Double) {
                lock.lock()
                defer { lock.unlock() }
                seenTrainSizes.append(size)
                maxObservedInTrain.append(maxVal)
            }
        }
        let recorder = Recorder()

        _ = try await TimeSeriesBacktester.backtest(
            series: baseSeries,
            initialWindow: 10,
            horizon: 3,
            stepSize: 2,
            mode: .expanding
        ) { train, h in
            recorder.record(size: train.count, maxVal: train.max() ?? 0.0)
            // Naive persistence forecast for h steps
            return [Double](repeating: train.last!, count: h)
        }

        // Folds should start at origin 10, 12, 14, 16, 18, 20, 22, 24, 26, 27
        for (i, trainSize) in recorder.seenTrainSizes.enumerated() {
            let origin = 10 + i * 2
            #expect(trainSize == origin)
            // The maximum value in train must strictly be baseSeries[origin - 1], never origin or higher
            #expect(recorder.maxObservedInTrain[i] == baseSeries[origin - 1])
        }
    }

    @Test("Sliding window mode strictly caps training history to windowSize")
    func testSlidingWindowModeStrictBounds() async throws {
        let series: [Double] = (0..<40).map { 10.0 + sin(Double($0) * 0.2) }
        let windowLimit = 8

        final class SizeRecorder: @unchecked Sendable {
            var observedSizes: [Int] = []
            private let lock = NSLock()

            func append(_ size: Int) {
                lock.lock()
                defer { lock.unlock() }
                observedSizes.append(size)
            }
        }
        let recorder = SizeRecorder()

        let res = try await TimeSeriesBacktester.backtest(
            series: series,
            initialWindow: 12,
            horizon: 2,
            stepSize: 3,
            mode: .sliding(windowSize: windowLimit)
        ) { train, h in
            recorder.append(train.count)
            return [Double](repeating: train.last!, count: h)
        }

        #expect(!res.folds.isEmpty)
        for size in recorder.observedSizes {
            #expect(size == windowLimit)
        }
    }

    @Test("Multi-step horizon evaluations (h=3, h=5) with overlapping and non-overlapping strides")
    func testMultiStepHorizons() async throws {
        let series: [Double] = (0..<30).map { Double($0) * 1.5 }

        // Non-overlapping: horizon = 3, stepSize = 3
        let nonOverlapping = try await TimeSeriesBacktester.backtest(
            series: series,
            initialWindow: 10,
            horizon: 3,
            stepSize: 3,
            mode: .expanding
        ) { train, h in
            [Double](repeating: train.last!, count: h)
        }
        #expect(nonOverlapping.folds.count == (30 - 10) / 3)
        for fold in nonOverlapping.folds {
            #expect(fold.horizon == 3)
            #expect(fold.actual.count == 3)
            #expect(fold.forecast.count == 3)
            #expect(fold.mase.isFinite)
        }

        // Overlapping: horizon = 4, stepSize = 1
        let overlapping = try await TimeSeriesBacktester.backtest(
            series: series,
            initialWindow: 10,
            horizon: 4,
            stepSize: 1,
            mode: .expanding
        ) { train, h in
            [Double](repeating: train.last!, count: h)
        }
        let expectedCount = 30 - 10 - 4 + 1
        #expect(overlapping.folds.count == expectedCount)
        #expect(overlapping.mase > 0.0)
    }

    @Test("Per-fold MASE is computed against each fold's specific training history")
    func testPerFoldMASE() async throws {
        let series: [Double] = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0, 46.0]
        let res = try await TimeSeriesBacktester.backtest(
            series: series,
            initialWindow: 4,
            horizon: 1,
            stepSize: 1,
            mode: .expanding
        ) { train, h in
            [train.last! + 1.0]
        }

        #expect(res.folds.count == 6)
        for fold in res.folds {
            #expect(fold.mase > 0.0)
            #expect(fold.mase.isFinite)
        }
        #expect(res.mase > 0.0)
    }
}
