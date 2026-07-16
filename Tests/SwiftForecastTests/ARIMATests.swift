import Testing
import Foundation
@testable import SwiftForecast

@Suite("ARIMA Model Tests")
struct ARIMATests {
    
    @Test("ARIMA(0, 0, 0) on random noise")
    func testARIMA000() async throws {
        let noise = [1.1, 0.9, 1.2, 0.8, 1.0, 1.1, 0.9, 1.0]
        let model = try ARIMAModel(p: 0, d: 0, q: 0)
        
        try await model.fit(series: noise)
        let result = try await model.forecast(horizon: 3)
        
        // Predictions should equal the mean of the series
        let mean = noise.reduce(0.0, +) / Double(noise.count)
        #expect(result.forecast.predictions.count == 3)
        for pred in result.forecast.predictions {
            #expect(abs(pred - mean) < 1e-7)
        }
    }
    
    @Test("AR(1) parameter recovery")
    func testAR1Recovery() async throws {
        // Generate AR(1) process with phi = 0.8, c = 2.0
        let phi = 0.8
        let c = 2.0
        var series = [Double](repeating: 0.0, count: 50)
        series[0] = 10.0
        var rng = SimpleRNG(seed: 42)
        for t in 1..<50 {
            series[t] = c + phi * series[t-1] + rng.nextGaussian() * 0.1
        }
        
        let model = try ARIMAModel(p: 1, d: 0, q: 0)
        try await model.fit(series: series)
        
        let result = try await model.forecast(horizon: 2)
        
        #expect(result.arCoefficients.count == 1)
        // Recovered coefficient should be close to 0.8 (e.g., between 0.65 and 0.95)
        #expect(result.arCoefficients[0] > 0.65 && result.arCoefficients[0] < 0.95)
        
        // Recovered intercept should be positive
        #expect(result.intercept > 0.0)
        
        // Check AIC calculation
        let aic = try await model.aic()
        #expect(!aic.isNaN)
    }
    @Test("ARIMA(1, 1, 0) on random walk")
    func testARIMA110OnRandomWalk() async throws {
        // Random walk y_t = y_{t-1} + e_t
        var series = [Double](repeating: 0.0, count: 30)
        series[0] = 50.0
        var rng = SimpleRNG(seed: 10)
        for t in 1..<30 {
            series[t] = series[t-1] + rng.nextGaussian() * 0.5
        }
        
        let model = try ARIMAModel(p: 1, d: 1, q: 0)
        try await model.fit(series: series)
        
        let result = try await model.forecast(horizon: 4)
        
        #expect(result.forecast.predictions.count == 4)
        // Predictions should remain near the last level of random walk
        let lastObs = series.last!
        #expect(abs(result.forecast.predictions[0] - lastObs) < 5.0)
    }
    
    @Test("ARIMA validation parameters")
    func testARIMAValidation() throws {
        #expect(throws: ForecastError.self) {
            _ = try ARIMAModel(p: -1, d: 0, q: 0)
        }
        #expect(throws: ForecastError.self) {
            _ = try ARIMAModel(p: 0, d: -1, q: 0)
        }
        #expect(throws: ForecastError.self) {
            _ = try ARIMAModel(p: 0, d: 0, q: -1)
        }
    }

    @Test("ARIMAX with exogenous inputs regression")
    func testARIMAX() async throws {
        // Generate a series with a clear exogenous relationship:
        // y_t = 10.0 + 2.5 * x_t + arima_noise
        let n = 40
        var exog: [[Double]] = []
        var series: [Double] = []
        
        var rng = SimpleRNG(seed: 123)
        
        for t in 0..<n {
            let xVal = Double(t) * 0.1 + rng.nextGaussian() * 0.1
            exog.append([xVal])
            let noise = rng.nextGaussian() * 0.2
            let yVal = 10.0 + 2.5 * xVal + noise
            series.append(yVal)
        }
        
        let model = try ARIMAModel(p: 1, d: 0, q: 0)
        try await model.fit(series: series, exog: exog)
        
        let exogForecast = [[Double(n) * 0.1], [Double(n + 1) * 0.1]]
        let result = try await model.forecast(horizon: 2, exog: exogForecast)
        
        let expected0 = 10.0 + 2.5 * exogForecast[0][0]
        let expected1 = 10.0 + 2.5 * exogForecast[1][0]
        
        print("DEBUG ARIMAX exogCoefficients: \(result.exogCoefficients)")
        print("DEBUG ARIMAX predictions: \(result.forecast.predictions)")
        print("DEBUG ARIMAX expected0: \(expected0), expected1: \(expected1)")
        
        #expect(result.exogCoefficients.count == 1)
        // Exogenous coefficient should be close to 2.5
        #expect(abs(result.exogCoefficients[0] - 2.5) < 0.5)
        
        // Predictions should match the relationship
        #expect(result.forecast.predictions.count == 2)
        #expect(abs(result.forecast.predictions[0] - expected0) < 1.0)
        #expect(abs(result.forecast.predictions[1] - expected1) < 1.0)
    }
}
