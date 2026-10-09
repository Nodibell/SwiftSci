import Testing
import Foundation
@testable import SwiftForecast

@Suite("SARIMA Model Tests")
struct SARIMATests {

    @Test("SARIMA(0, 0, 0)(0, 0, 0)s on random noise")
    func testSARIMA000() async throws {
        let noise = [1.1, 0.9, 1.2, 0.8, 1.0, 1.1, 0.9, 1.0]
        let model = try SARIMAModel(p: 0, d: 0, q: 0, P: 0, D: 0, Q: 0, s: 4)

        try await model.fit(series: noise)
        let predictions = try await model.forecast(steps: 3)

        #expect(predictions.count == 3)
        let mean = noise.reduce(0.0, +) / Double(noise.count)
        for pred in predictions {
            #expect(abs(pred - mean) < 0.2)
        }
    }

    @Test("SARIMA parameter validation")
    func testSARIMAValidation() throws {
        #expect(throws: (any Error).self) {
            _ = try SARIMAModel(p: -1, d: 0, q: 0, P: 0, D: 0, Q: 0, s: 4)
        }
        #expect(throws: (any Error).self) {
            _ = try SARIMAModel(p: 0, d: 0, q: 0, P: 0, D: 0, Q: 0, s: 0)
        }
        #expect(throws: (any Error).self) {
            _ = try SARIMAModel(p: 0, d: -1, q: 0, P: 0, D: 0, Q: 0, s: 4)
        }
    }

    @Test("SARIMA rejects input series with NaN or Infinity")
    func testSARIMANaNAndInfinityValidation() async throws {
        let model = try SARIMAModel(p: 1, d: 0, q: 0, P: 0, D: 0, Q: 0, s: 4)
        let nanSeries = [1.0, 2.0, Double.nan, 4.0, 5.0, 6.0, 7.0, 8.0]
        await #expect(throws: ForecastError.self) {
            try await model.fit(series: nanSeries)
        }

        let infSeries = [1.0, 2.0, Double.infinity, 4.0, 5.0, 6.0, 7.0, 8.0]
        await #expect(throws: ForecastError.self) {
            try await model.fit(series: infSeries)
        }
    }

    @Test("SARIMA forecast invalid step parameter validation")
    func testSARIMAInvalidSteps() async throws {
        let series = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let model = try SARIMAModel(p: 1, d: 0, q: 0, P: 0, D: 0, Q: 0, s: 4)
        try await model.fit(series: series)

        await #expect(throws: ForecastError.self) {
            _ = try await model.forecast(steps: 0)
        }
        await #expect(throws: ForecastError.self) {
            _ = try await model.forecast(steps: -2)
        }
    }

    @Test("SARIMA multiplicative cross-term polynomial estimation")
    func testSARIMAMultiplicativeCrossTerms() async throws {
        // Generate true multiplicative SARIMA(1, 0, 0)(1, 0, 0)[4] process:
        // (1 - phi * B)(1 - Phi * B^4) y_t = e_t
        // y_t = phi * y_{t-1} + Phi * y_{t-4} - (phi * Phi) * y_{t-5} + e_t
        let phi = 0.5
        let Phi = 0.6
        let cross = phi * Phi // 0.30
        let s = 4
        let n = 80

        var series = [Double](repeating: 0.0, count: n)
        var rng = SimpleRNG(seed: 42)
        for t in 0..<5 {
            series[t] = 10.0 + rng.nextGaussian() * 0.1
        }

        for t in 5..<n {
            let val = phi * series[t - 1] + Phi * series[t - s] - cross * series[t - 1 - s] + rng.nextGaussian() * 0.1
            series[t] = val
        }

        let model = try SARIMAModel(p: 1, d: 0, q: 0, P: 1, D: 0, Q: 0, s: s)
        try await model.fit(series: series)

        let fittedAR = await model.arCoefficients
        let fittedSAR = await model.seasonalArCoefficients

        #expect(fittedAR.count == 1)
        #expect(fittedSAR.count == 1)

        // Estimated parameters should be close to true parameters (within reasonable tolerance)
        #expect(abs(fittedAR[0] - phi) < 0.25)
        #expect(abs(fittedSAR[0] - Phi) < 0.25)

        // Out-of-sample forecast
        let forecasts = try await model.forecast(steps: 4)
        #expect(forecasts.count == 4)
        for f in forecasts {
            #expect(f.isFinite && !f.isNaN)
        }
    }

    @Test("SARIMA seasonal and non-seasonal differencing integration")
    func testSARIMASeasonalDifferencingIntegration() async throws {
        // Linear trend + seasonal pattern + noise with (d=1, D=1, s=4)
        let s = 4
        let n = 40
        var series = [Double](repeating: 0.0, count: n)
        let seasonals = [5.0, -2.0, 8.0, -4.0]

        for t in 0..<n {
            series[t] = 10.0 + 2.0 * Double(t) + seasonals[t % s]
        }

        let model = try SARIMAModel(p: 0, d: 1, q: 0, P: 0, D: 1, Q: 0, s: s)
        try await model.fit(series: series)

        let forecasts = try await model.forecast(steps: 4)
        #expect(forecasts.count == 4)

        // Forecasts should continue upward trend with seasonal shape
        #expect(forecasts[0] > series.last!)
        for f in forecasts {
            #expect(f.isFinite)
        }
    }

    @Test("SARIMA AIC and BIC computation")
    func testSARIMAAICandBIC() async throws {
        let series = (0..<40).map { Double($0) * 0.5 + Double($0 % 4) }
        let model = try SARIMAModel(p: 1, d: 0, q: 0, P: 1, D: 0, Q: 0, s: 4)
        try await model.fit(series: series)

        let aic = try await model.aic()
        let bic = try await model.bic()

        #expect(aic.isFinite && !aic.isNaN)
        #expect(bic.isFinite && !bic.isNaN)
        // For N=35 > e^2 ~= 7.39, log(N) > 2, hence BIC > AIC
        #expect(bic > aic)
    }
}
