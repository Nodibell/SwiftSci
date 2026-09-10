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

        let autoArima = AutoARIMA(maxP: 2, maxD: 1, maxQ: 2, criterion: .aic)
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

        let autoArima = AutoARIMA(maxP: 2, maxD: 2, maxQ: 2, d: 1, criterion: .bic)
        _ = try await autoArima.fit(series: series)

        let bestOrder = await autoArima.bestOrder
        #expect(bestOrder != nil)
        #expect(bestOrder!.d == 1) // Fixed d was respected
    }

    @Test("AutoARIMA input validation and un-fitted errors")
    func testAutoARIMAValidation() async {
        let autoArima = AutoARIMA()
        await #expect(throws: ForecastError.self) {
            _ = try await autoArima.fit(series: [])
        }
        await #expect(throws: ForecastError.self) {
            _ = try await autoArima.forecast(horizon: 5)
        }
    }
}
