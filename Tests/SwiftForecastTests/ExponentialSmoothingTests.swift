import Testing
import Foundation
@testable import SwiftForecast

@Suite("Exponential Smoothing Tests")
struct ExponentialSmoothingTests {
    
    @Test("Simple Exponential Smoothing on constant series")
    func testSESConstantSeries() async throws {
        let series = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        let model = ExponentialSmoothing(method: .simple, alpha: 0.5)
        
        try await model.fit(series: series)
        let forecast = try await model.forecast(horizon: 3)
        
        #expect(forecast.predictions.count == 3)
        for pred in forecast.predictions {
            #expect(abs(pred - 5.0) < 1e-7)
        }
        
        #expect(forecast.mse == 0.0)
    }
    
    @Test("Simple Exponential Smoothing with parameter optimization")
    func testSESParameterOptimization() async throws {
        let series = [1.0, 3.0, 2.0, 4.0, 3.0, 5.0]
        // Fit with alpha = nil to trigger grid search
        let model = ExponentialSmoothing(method: .simple, alpha: nil)
        
        try await model.fit(series: series)
        let forecast = try await model.forecast(horizon: 2)
        
        #expect(forecast.predictions.count == 2)
        // Check that mse is calculated
        #expect(forecast.mse > 0.0)
    }
    
    @Test("Holt's Double Exponential Smoothing on linear trend series")
    func testHoltLinearTrend() async throws {
        // Perfectly linear series: y = 2 * t + 3
        var series: [Double] = []
        for t in 0..<10 {
            series.append(2.0 * Double(t) + 3.0)
        }
        
        let model = ExponentialSmoothing(method: .double(beta: 0.2), alpha: 0.8)
        try await model.fit(series: series)
        
        let forecast = try await model.forecast(horizon: 3)
        
        // Predictions should continue the trend:
        // series[9] = 21.0. Next steps should be 23.0, 25.0, 27.0
        #expect(forecast.predictions.count == 3)
        #expect(abs(forecast.predictions[0] - 23.0) < 0.5)
        #expect(abs(forecast.predictions[1] - 25.0) < 0.5)
        #expect(abs(forecast.predictions[2] - 27.0) < 0.5)
    }
    
    @Test("Holt-Winters additive seasonality forecast")
    func testHoltWintersAdditive() async throws {
        // period = 4
        // trend = 0.5 * t
        // seasonal = [1.0, -2.0, 3.0, -2.0]
        let seasonalPattern = [1.0, -2.0, 3.0, -2.0]
        var series: [Double] = []
        for t in 0..<40 {
            let trendVal = Double(t) * 0.5
            let seasonalVal = seasonalPattern[t % 4]
            series.append(trendVal + seasonalVal)
        }
        
        let model = ExponentialSmoothing(
            method: .holtWinters(beta: 0.1, gamma: 0.1, period: 4, seasonal: .additive),
            alpha: 0.3
        )
        try await model.fit(series: series)
        
        let forecast = try await model.forecast(horizon: 4)
        #expect(forecast.predictions.count == 4)
        
        // Expected future values for t = 40, 41, 42, 43
        // predictions should combine the linear trend continuation and seasonal pattern:
        // t = 40: trend = 20.0, seasonal = 1.0 -> y_hat = 21.0
        // t = 41: trend = 20.5, seasonal = -2.0 -> y_hat = 18.5
        // t = 42: trend = 21.0, seasonal = 3.0 -> y_hat = 24.0
        // t = 43: trend = 21.5, seasonal = -2.0 -> y_hat = 19.5
        #expect(abs(forecast.predictions[0] - 21.0) < 0.5)
        #expect(abs(forecast.predictions[1] - 18.5) < 0.5)
        #expect(abs(forecast.predictions[2] - 24.0) < 0.5)
        #expect(abs(forecast.predictions[3] - 19.5) < 0.5)
    }
    
    @Test("Holt-Winters validation edge cases")
    func testHWValidation() async throws {
        let model = ExponentialSmoothing(method: .holtWinters(beta: 0.1, gamma: 0.1, period: 4, seasonal: .additive))
        
        // Insufficient length (period * 2 = 8 required)
        let shortSeries = [1.0, 2.0, 3.0, 4.0, 5.0]
        await #expect(throws: ForecastError.self) {
            try await model.fit(series: shortSeries)
        }
        
        // Call forecast before fit
        let unfittedModel = ExponentialSmoothing(method: .simple)
        await #expect(throws: ForecastError.self) {
            try await unfittedModel.forecast(horizon: 2)
        }
    }
}
