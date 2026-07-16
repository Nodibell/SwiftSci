import Testing
import Foundation
import Accelerate
@testable import SwiftForecast

@Suite("Time Series Decomposition Tests")
struct DecompositionTests {
    
    @Test("Classical additive decomposition on clean trend + seasonal + noise")
    func testAdditiveDecomposition() throws {
        // period = 4
        // trend: t * 0.5
        // seasonal: [1.0, -2.0, 3.0, -2.0]
        let seasonalPattern = [1.0, -2.0, 3.0, -2.0]
        var series: [Double] = []
        for t in 0..<16 {
            let trendVal = Double(t) * 0.5
            let seasonalVal = seasonalPattern[t % 4]
            series.append(trendVal + seasonalVal)
        }
        
        let result = try TimeSeriesDecomposition.decompose(series: series, period: 4, model: .additive)
        
        // Check array sizes
        #expect(result.trend.count == 16)
        #expect(result.seasonal.count == 16)
        #expect(result.residual.count == 16)
        
        // Edge elements of trend should be NaN (2 elements at each side for period=4 centered MA)
        #expect(result.trend[0].isNaN)
        #expect(result.trend[1].isNaN)
        #expect(result.trend[14].isNaN)
        #expect(result.trend[15].isNaN)
        
        // Center values of trend should be close to actual trend values
        for i in 2...13 {
            let expectedTrend = Double(i) * 0.5
            #expect(abs(result.trend[i] - expectedTrend) < 1e-7)
        }
        
        // Check seasonal pattern matching (should sum to 0)
        let cycle = Array(result.seasonal.prefix(4))
        let cycleSum = cycle.reduce(0.0, +)
        #expect(abs(cycleSum) < 1e-7)
        
        // Check residuals are close to 0 (since noise was 0)
        for i in 2...13 {
            #expect(abs(result.residual[i]) < 1e-7)
        }
    }
    
    @Test("Classical multiplicative decomposition")
    func testMultiplicativeDecomposition() throws {
        let seasonalPattern = [1.2, 0.8, 1.5, 0.5]
        var series: [Double] = []
        for t in 0..<16 {
            let trendVal = 10.0 + Double(t) * 0.2
            let seasonalVal = seasonalPattern[t % 4]
            series.append(trendVal * seasonalVal)
        }
        
        let result = try TimeSeriesDecomposition.decompose(series: series, period: 4, model: .multiplicative)
        
        // Center values of residuals should be close to 1.0 (no noise)
        for i in 2...13 {
            #expect(abs(result.residual[i] - 1.0) < 0.02)
        }
        
        // Seasonal cycle mean should be close to 1.0
        let cycle = Array(result.seasonal.prefix(4))
        let cycleMean = vDSP.mean(cycle)
        #expect(abs(cycleMean - 1.0) < 1e-7)
    }
    
    @Test("Decomposition validation and edge cases")
    func testDecompositionValidation() throws {
        let shortSeries = [1.0, 2.0, 3.0]
        #expect(throws: ForecastError.self) {
            try TimeSeriesDecomposition.decompose(series: shortSeries, period: 4)
        }
        
        let emptySeries: [Double] = []
        #expect(throws: ForecastError.self) {
            try TimeSeriesDecomposition.decompose(series: emptySeries, period: 2)
        }
        
        let nanSeries = [1.0, Double.nan, 3.0, 4.0, 5.0, 6.0]
        #expect(throws: ForecastError.self) {
            try TimeSeriesDecomposition.decompose(series: nanSeries, period: 2)
        }
    }
    
    @Test("Autocorrelation function (ACF)")
    func testACF() throws {
        let series = [1.0, 2.0, 3.0, 4.0, 5.0]
        let result = try TimeSeriesDecomposition.acf(series: series, maxLag: 2)
        #expect(result.count == 3)
        #expect(result[0] == 1.0) // Lag 0 correlation is always 1.0
        #expect(result[1] < 1.0)
    }
    
    @Test("Partial Autocorrelation function (PACF) via Yule-Walker")
    func testPACF() throws {
        let series = [1.0, 2.0, 1.5, 2.5, 2.0, 3.0]
        let result = try TimeSeriesDecomposition.pacf(series: series, maxLag: 2)
        #expect(result.count == 3)
        #expect(result[0] == 1.0)
    }
    
    @Test("Augmented Dickey-Fuller (ADF) test stationarity calculations")
    func testADFTest() throws {
        // Stationary series (mean-reverting around 5)
        let stationary: [Double] = [
            5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9, 5.2, 4.8, 5.0,
            5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9, 5.2, 4.8, 5.0
        ]
        let (statStat, statP) = try TimeSeriesDecomposition.adfTest(series: stationary, maxLag: 1)
        // Stationary series should have low (negative) statistic and low p-value
        #expect(statStat < 0.0)
        #expect(statP < 0.3)

        // Non-stationary random walk series
        var randomWalk = [Double](repeating: 0.0, count: 25)
        randomWalk[0] = 100.0
        var rng = SimpleRNG(seed: 42)
        for i in 1..<25 {
            randomWalk[i] = randomWalk[i-1] + rng.nextGaussian()
        }
        let (_, rwP) = try TimeSeriesDecomposition.adfTest(series: randomWalk, maxLag: 1)
        
        // Unit root series should have high p-value (fail to reject non-stationarity)
        #expect(rwP > 0.05)
    }
}
