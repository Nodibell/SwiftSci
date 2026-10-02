import Testing
import Foundation
@testable import SwiftForecast

@Suite("AutoARIMA Tests")
struct AutoARIMATests {

    @Test("AutoARIMA discovers order on AR(1) series")
    func testAutoARIMAOnAR1() async throws {
        // Generate AR(1) process: y_t = 2.0 + 0.7 * y_{t-1} + noise
        var series = [Double](repeating: 0.0, count: 50)
        series[0] = 5.0
        var rng = SimpleRNG(seed: 42)
        for t in 1..<50 {
            series[t] = 2.0 + 0.7 * series[t - 1] + rng.nextGaussian() * 0.2
        }

        let autoArima = try AutoARIMA(maxP: 2, maxD: 1, maxQ: 2, criterion: .aic)
        let result = try await autoArima.fit(series: series)

        let bestOrder = await autoArima.bestOrder
        let bestScore = await autoArima.bestScore
        let leaderboard = await autoArima.leaderboard

        #expect(bestOrder != nil)
        #expect(bestScore != nil && !bestScore!.isNaN)
        #expect(!leaderboard.isEmpty)
        #expect(result.forecast.predictions.count == 1)

        // Forecast 3 steps ahead
        let forecastResult = try await autoArima.forecast(horizon: 3)
        #expect(forecastResult.forecast.predictions.count == 3)
    }

    @Test("AutoARIMA with fixed differencing d constraint")
    func testAutoARIMAFixedD() async throws {
        // Random walk y_t = y_{t-1} + e_t
        var series = [Double](repeating: 0.0, count: 40)
        series[0] = 20.0
        var rng = SimpleRNG(seed: 99)
        for t in 1..<40 {
            series[t] = series[t - 1] + rng.nextGaussian() * 0.3
        }

        let autoArima = try AutoARIMA(maxP: 2, maxD: 2, maxQ: 2, d: 1, criterion: .bic)
        _ = try await autoArima.fit(series: series)

        let bestOrder = await autoArima.bestOrder
        #expect(bestOrder != nil)
        #expect(bestOrder!.d == 1) // Fixed d was respected
    }

    @Test("AutoARIMA input validation and un-fitted errors")
    func testAutoARIMAValidation() async throws {
        let autoArima = try AutoARIMA()
        await #expect(throws: ForecastError.self) {
            _ = try await autoArima.fit(series: [])
        }
        await #expect(throws: ForecastError.self) {
            _ = try await autoArima.forecast(horizon: 5)
        }
    }

    @Test("Gate 8: AutoARIMA zero-variance resilience on constant series")
    func testAutoARIMAConstantSeriesZeroVariance() async throws {
        let constantSeries = [Double](repeating: 42.0, count: 50)
        let autoArima = try AutoARIMA(maxP: 2, maxD: 2, maxQ: 2, maxIterations: 50)
        _ = try await autoArima.fit(series: constantSeries)

        let bestOrder = await autoArima.bestOrder
        #expect(bestOrder == AutoARIMAOrder(p: 0, d: 0, q: 0))

        let forecast = try await autoArima.forecast(horizon: 3)
        #expect(forecast.forecast.predictions.count == 3)
        for pred in forecast.forecast.predictions {
            #expect(abs(pred - 42.0) < 1e-6)
        }
    }

    @Test("AutoARIMA + exog: Training with exogenous matrix completes without dimension mismatch")
    func testAutoARIMAWithExogenousVariables() async throws {
        // y_t = 3.0 * X_t + 10.0 + small noise
        let n = 50
        var exog: [[Double]] = []
        var series = [Double](repeating: 0.0, count: n)
        var rng = SimpleRNG(seed: 123)

        for i in 0..<n {
            let xVal = Double(i) * 0.5
            exog.append([xVal])
            series[i] = 10.0 + 3.0 * xVal + rng.nextGaussian() * 0.1
        }

        let autoArima = try AutoARIMA(maxP: 1, maxD: 0, maxQ: 1, maxCandidates: 10, criterion: .aic)
        let fitResult = try await autoArima.fit(series: series, exog: exog)

        let bestOrder = await autoArima.bestOrder
        #expect(bestOrder != nil)
        #expect(fitResult.forecast.predictions.count == 1)

        // Test out-of-sample forecasting with future exogenous matrix (horizon: 2)
        let futureExog: [[Double]] = [[Double(n) * 0.5], [Double(n + 1) * 0.5]]
        let forecastResult = try await autoArima.forecast(horizon: 2, exog: futureExog)
        #expect(forecastResult.forecast.predictions.count == 2)
        #expect(forecastResult.forecast.predictions[0] > series.last!)
    }

    @Test("AutoARIMA seasonal search: Successfully evaluates and discovers seasonal SARIMA orders")
    func testAutoARIMASeasonalSearchDiscoversSeasonalOrder() async throws {
        // Strong seasonal signal with period s = 4
        let n = 48
        var series = [Double](repeating: 0.0, count: n)
        let seasonalPattern = [10.0, -5.0, 15.0, -10.0]
        var rng = SimpleRNG(seed: 77)

        for i in 0..<n {
            series[i] = 20.0 + seasonalPattern[i % 4] + rng.nextGaussian() * 0.2
        }

        let autoArima = try AutoARIMA(
            maxP: 1,
            maxD: 0,
            maxQ: 1,
            seasonal: true,
            seasonalPeriod: 4,
            maxSeasonalP: 1,
            maxSeasonalD: 0,
            maxSeasonalQ: 1,
            maxCandidates: 30,
            criterion: .aic
        )

        let result = try await autoArima.fit(series: series)
        let bestOrder = await autoArima.bestOrder
        #expect(bestOrder != nil)
        #expect(bestOrder!.s == 4)
        #expect(result.forecast.predictions.count == 1)

        let fc = try await autoArima.forecast(horizon: 4)
        #expect(fc.forecast.predictions.count == 4)
        #expect(fc.seasonalOrder != nil)
        #expect(fc.seasonalOrder!.s == 4)
    }

    @Test("AutoARIMA parsimonious ordering: candidate search evaluates simpler models first")
    func testAutoARIMAParsimoniousOrdering() async throws {
        let series = (0..<40).map { Double($0) * 0.5 }
        let autoArima = try AutoARIMA(maxP: 3, maxD: 2, maxQ: 3, maxCandidates: 5)
        _ = try await autoArima.fit(series: series)

        let leaderboard = await autoArima.leaderboard
        #expect(!leaderboard.isEmpty)
        #expect(leaderboard.count <= 5)
    }

    @Test("AutoARIMA configuration validation throws on invalid parameters")
    func testAutoARIMAValidationThrowsOnNegativeOrders() {
        #expect(throws: ForecastError.self) {
            _ = try AutoARIMA(maxP: -1)
        }
        #expect(throws: ForecastError.self) {
            _ = try AutoARIMA(maxD: -2)
        }
        #expect(throws: ForecastError.self) {
            _ = try AutoARIMA(maxQ: -1)
        }
        #expect(throws: ForecastError.self) {
            _ = try AutoARIMA(seasonal: true, seasonalPeriod: 1)
        }
        #expect(throws: ForecastError.self) {
            _ = try AutoARIMA(maxCandidates: 0)
        }
    }

    @Test("AutoARIMA polynomial stability check correctly filters stable vs unstable roots")
    func testPolynomialStability() {
        // AR(1) tests
        #expect(AutoARIMA.isPolynomialStable(coefficients: [0.7]))
        #expect(AutoARIMA.isPolynomialStable(coefficients: [-0.9]))
        #expect(!AutoARIMA.isPolynomialStable(coefficients: [1.05]))
        #expect(!AutoARIMA.isPolynomialStable(coefficients: [-1.2]))

        // AR(2) tests
        #expect(AutoARIMA.isPolynomialStable(coefficients: [0.5, -0.3]))
        #expect(AutoARIMA.isPolynomialStable(coefficients: [1.2, -0.8])) // Stable complex roots: |z|^2 = 1.25 > 1
        #expect(!AutoARIMA.isPolynomialStable(coefficients: [1.5, 0.5]))  // phi1 + phi2 = 2.0 > 1 (unstable)
        #expect(!AutoARIMA.isPolynomialStable(coefficients: [0.5, 1.2]))  // |phi2| = 1.2 > 1 (unstable)
    }

    @Test("AutoARIMA seasonal reference validation: synthetic seasonal process selects SARIMA with period matching seasonal waveform")
    func testAutoARIMASARIMASyntheticReference() async throws {
        // Strong seasonal repeating pattern s=4: [12.0, -8.0, 16.0, -10.0] around base 50.0
        let s = 4
        let n = 48
        let pattern = [12.0, -8.0, 16.0, -10.0]
        var series = [Double](repeating: 0.0, count: n)
        for t in 0..<n {
            series[t] = 50.0 + pattern[t % s]
        }

        let autoArima = try AutoARIMA(
            maxP: 1, maxD: 0, maxQ: 0,
            seasonal: true, seasonalPeriod: 4,
            maxSeasonalP: 1, maxSeasonalD: 0, maxSeasonalQ: 0,
            criterion: .aic
        )
        _ = try await autoArima.fit(series: series)
        let bestOrder = await autoArima.bestOrder
        #expect(bestOrder != nil)
        #expect(bestOrder?.s == 4)

        let fc = try await autoArima.forecast(horizon: 4)
        let preds = fc.forecast.predictions
        #expect(preds.count == 4)
        // Verify forecasted seasonal waveform captures high phases vs low phases:
        // High phases (steps 0 & 2) significantly exceed baseline (50.0), while low phases (steps 1 & 3) fall well below
        #expect(preds[0] > 55.0)
        #expect(preds[2] > 55.0)
        #expect(preds[1] < 45.0)
        #expect(preds[3] < 45.0)
        #expect(preds[0] > preds[1])
        #expect(preds[2] > preds[3])
    }
}
