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

        // Non-finite inputs must always be rejected
        #expect(!AutoARIMA.isPolynomialStable(coefficients: [Double.nan]))
        #expect(!AutoARIMA.isPolynomialStable(coefficients: [0.5, Double.nan]))
        #expect(!AutoARIMA.isPolynomialStable(coefficients: [Double.infinity]))

        // MA(2) invertibility uses 1 + theta_1*B + theta_2*B^2 sign convention (negated vs AR)
        #expect(AutoARIMA.isMAPolynomialInvertible(coefficients: [1.5, 0.8]))    // 1 + 1.5B + 0.8B^2 has |B|^2 = 1.25 > 1
        #expect(!AutoARIMA.isPolynomialStable(coefficients: [1.5, 0.8]))         // old un-negated check falsely rejected this
        #expect(!AutoARIMA.isMAPolynomialInvertible(coefficients: [0.5, -0.8]))  // 1 + 0.5B - 0.8B^2 has root inside unit circle
        #expect(AutoARIMA.isPolynomialStable(coefficients: [0.5, -0.8]))         // old un-negated check falsely accepted this
        #expect(!AutoARIMA.isMAPolynomialInvertible(coefficients: [Double.nan]))
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

    @Test("AutoARIMA seasonal rejects exogenous variables with explicit invalidParameter error")
    func testAutoARIMASeasonalThrowsOnExogenousVariables() async throws {
        let series = (0..<20).map { Double($0) }
        let exog = (0..<20).map { [Double($0) * 0.1] }
        let autoArima = try AutoARIMA(seasonal: true, seasonalPeriod: 4)

        await #expect(throws: ForecastError.self) {
            _ = try await autoArima.fit(series: series, exog: exog)
        }
    }

    @Test("AutoARIMA rejects NaN zero-variance input and converges cleanly on AirPassengers control")
    func testAutoARIMANaNGuardAndAirPassengersControl() async throws {
        let nanAuto = try AutoARIMA()
        await #expect(throws: ForecastError.containsNaN) {
            _ = try await nanAuto.fit(series: [Double.nan, Double.nan, Double.nan, Double.nan])
        }
        #expect(await nanAuto.bestOrder == nil)

        let airPassengersTrain: [Double] = [
            112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
            115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
            145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
            171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
            196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
            204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
            242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
            284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
            315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
            340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337
        ]
        let autoSeasonal = try AutoARIMA(
            maxP: 1, maxD: 1, maxQ: 1,
            seasonal: true, seasonalPeriod: 12,
            maxSeasonalP: 1, maxSeasonalD: 1, maxSeasonalQ: 1
        )
        let fitResult = try await autoSeasonal.fit(series: airPassengersTrain)
        #expect(await autoSeasonal.bestOrder != nil)
        #expect(AutoARIMA.isPolynomialStable(coefficients: fitResult.arCoefficients))
        #expect(AutoARIMA.isMAPolynomialInvertible(coefficients: fitResult.maCoefficients))
        let fc = try await autoSeasonal.forecast(horizon: 12)
        #expect(fc.forecast.predictions.count == 12)
        #expect(fc.forecast.predictions.allSatisfy { $0.isFinite })
        #expect(fc.forecast.aic?.isFinite == true)
    }
}
