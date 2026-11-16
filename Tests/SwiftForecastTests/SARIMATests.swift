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

    private func makeSyntheticSeasonalUniformSeries(count: Int = 500, seed: UInt64 = 42) -> [Double] {
        var rng = BenchmarkLCG(seed: seed)
        var result: [Double] = []
        result.reserveCapacity(count)
        for t in 0..<count {
            let dt = Double(t)
            let angle: Double = (2.0 * Double.pi * dt) / 12.0
            let trend: Double = 100.0 + 0.2 * dt
            let seasonal: Double = 8.0 * sin(angle)
            let noise: Double = rng.nextDouble(in: -1.5...1.5)
            let value: Double = (trend + seasonal) + noise
            result.append(value)
        }
        return result
    }

    private func makeSyntheticSeasonalGaussianSeries(count: Int = 50_000, seed: UInt64 = 42) -> [Double] {
        var rng = BenchmarkLCG(seed: seed)
        var result: [Double] = []
        result.reserveCapacity(count)
        for t in 0..<count {
            let dt = Double(t)
            let angle: Double = (2.0 * Double.pi * dt) / 12.0
            let trend: Double = 100.0 + 0.002 * dt
            let seasonal: Double = 10.0 * sin(angle)
            let noise: Double = rng.nextGaussian() * 0.8
            let value: Double = (trend + seasonal) + noise
            result.append(value)
        }
        return result
    }

    private func computeRMSE(predicted: [Double], actual: [Double]) -> Double {
        precondition(predicted.count == actual.count && !predicted.isEmpty)
        var sumSquaredError: Double = 0.0
        for i in 0..<predicted.count {
            let diff: Double = predicted[i] - actual[i]
            sumSquaredError += diff * diff
        }
        let meanSquaredError: Double = sumSquaredError / Double(predicted.count)
        return sqrt(meanSquaredError)
    }

    @Test("SARIMA rejects non-invertible syn500/syn50k (1,1,1)(1,0,0)[12] and preserves finite (1,1,1)(0,1,0)[12] forecasts")
    func testSARIMASyntheticStabilityAndInvertibility() async throws {
        let syn500: [Double] = makeSyntheticSeasonalUniformSeries(count: 500, seed: 42)
        let train500 = Array(syn500.prefix(476))
        let test500 = Array(syn500.suffix(24))

        let bad500 = try SARIMAModel(p: 1, d: 1, q: 1, P: 1, D: 0, Q: 0, s: 12)
        await #expect(throws: ForecastError.self) {
            try await bad500.fit(series: train500)
        }
        #expect(await !bad500.isFitted)
        await #expect(throws: ForecastError.notFitted) {
            _ = try await bad500.forecast(steps: 1)
        }
        await #expect(throws: ForecastError.notFitted) {
            _ = try await bad500.aic()
        }

        let good500 = try SARIMAModel(p: 1, d: 1, q: 1, P: 0, D: 1, Q: 0, s: 12)
        try await good500.fit(series: train500)
        #expect(await good500.isFitted)
        let fc500 = try await good500.forecast(steps: 24)
        #expect(fc500.count == 24)
        #expect(fc500.allSatisfy { $0.isFinite })
        let rmse500: Double = computeRMSE(predicted: fc500, actual: test500)
        #expect(rmse500 < 2.0)
        #expect(try await good500.aic().isFinite)
        #expect(try await good500.bic().isFinite)

        let syn50k: [Double] = makeSyntheticSeasonalGaussianSeries(count: 50_000, seed: 42)
        let train50k = Array(syn50k.prefix(49_976))
        let test50k = Array(syn50k.suffix(24))

        let bad50k = try SARIMAModel(p: 1, d: 1, q: 1, P: 1, D: 0, Q: 0, s: 12)
        await #expect(throws: ForecastError.self) {
            try await bad50k.fit(series: train50k)
        }
        #expect(await !bad50k.isFitted)

        let good50k = try SARIMAModel(p: 1, d: 1, q: 1, P: 0, D: 1, Q: 0, s: 12)
        try await good50k.fit(series: train50k)
        #expect(await good50k.isFitted)
        let fc50k = try await good50k.forecast(steps: 24)
        #expect(fc50k.allSatisfy { $0.isFinite })
        let rmse50k: Double = computeRMSE(predicted: fc50k, actual: test50k)
        #expect(rmse50k < 2.5)
    }

    @Test("SARIMA preserves stable AirPassengers controls and clears isFitted on failed refit")
    func testSARIMAAirPassengersControlAndRefitReset() async throws {
        let airPassengers: [Double] = [
            112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
            115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
            145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
            171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
            196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
            204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
            242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
            284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
            315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
            340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
            360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
            417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
        ]
        let train = Array(airPassengers.prefix(120))

        // Control 1: SARIMA(1,1,1)(1,0,0)[12] must preserve valid existing predictions on AirPassengers
        let apModel1 = try SARIMAModel(p: 1, d: 1, q: 1, P: 1, D: 0, Q: 0, s: 12)
        try await apModel1.fit(series: train)
        #expect(await apModel1.isFitted)
        let fc1 = try await apModel1.forecast(steps: 24)
        #expect(fc1.count == 24)
        #expect(fc1.allSatisfy { $0.isFinite })
        #expect(abs(fc1[0] - 343.4956335276581) < 1e-8)
        #expect(abs(fc1[1] - 319.71685686831023) < 1e-8)
        #expect(abs(fc1[2] - 366.9356604863019) < 1e-8)

        // Control 2: Canonical airline model SARIMA(0,1,1)(0,1,1)[12]
        let apModel2 = try SARIMAModel(p: 0, d: 1, q: 1, P: 0, D: 1, Q: 1, s: 12)
        try await apModel2.fit(series: train)
        #expect(await apModel2.isFitted)
        let fc2 = try await apModel2.forecast(steps: 24)
        #expect(fc2.allSatisfy { $0.isFinite })
        #expect(abs(fc2[0] - 342.7946704250881) < 1e-8)
        #expect(abs(fc2[1] - 321.4323556998727) < 1e-8)
        #expect(abs(fc2[2] - 366.15239606803044) < 1e-8)

        // Failed refit must reset previously fitted model back to isFitted == false
        let syn500: [Double] = makeSyntheticSeasonalUniformSeries(count: 500, seed: 42)
        await #expect(throws: ForecastError.self) {
            try await apModel1.fit(series: Array(syn500.prefix(476)))
        }
        #expect(await !apModel1.isFitted)
        await #expect(throws: ForecastError.notFitted) {
            _ = try await apModel1.forecast(steps: 1)
        }
    }
}

private struct BenchmarkLCG {
    private var state: UInt64
    private var hasSpare = false
    private var spare = 0.0

    init(seed: UInt64 = 42) {
        self.state = seed
    }

    mutating func nextUniform() -> Double {
        state = 6364136223846793005 &* state &+ 1442695040888963407
        return Double(state >> 11) / Double(1 << 53)
    }

    mutating func nextDouble(in range: ClosedRange<Double>) -> Double {
        range.lowerBound + (range.upperBound - range.lowerBound) * nextUniform()
    }

    mutating func nextGaussian() -> Double {
        if hasSpare {
            hasSpare = false
            return spare
        }
        var u1 = nextUniform()
        while u1 <= 1e-12 {
            u1 = nextUniform()
        }
        let u2 = nextUniform()
        let mag = sqrt(-2.0 * log(u1))
        spare = mag * sin(2.0 * .pi * u2)
        hasSpare = true
        return mag * cos(2.0 * .pi * u2)
    }
}
