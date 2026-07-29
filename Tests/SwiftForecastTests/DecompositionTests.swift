import Testing
import Foundation
import Accelerate
@testable import SwiftForecast

@Suite("Time Series Decomposition Tests")
struct DecompositionTests {
    
    @Test("Classical additive decomposition on clean trend + seasonal + noise")
    func testAdditiveDecomposition() throws {
        let seasonalPattern = [1.0, -2.0, 3.0, -2.0]
        var series: [Double] = []
        for t in 0..<16 {
            let trendVal = Double(t) * 0.5
            let seasonalVal = seasonalPattern[t % 4]
            series.append(trendVal + seasonalVal)
        }
        
        let result = try TimeSeriesDecomposition.decompose(series: series, period: 4, model: .additive)
        
        #expect(result.trend.count == 16)
        #expect(result.seasonal.count == 16)
        #expect(result.residual.count == 16)
        
        #expect(result.trend[0].isNaN)
        #expect(result.trend[1].isNaN)
        #expect(result.trend[14].isNaN)
        #expect(result.trend[15].isNaN)
        
        for i in 2...13 {
            let expectedTrend = Double(i) * 0.5
            #expect(abs(result.trend[i] - expectedTrend) < 1e-7)
        }
        
        let cycle = Array(result.seasonal.prefix(4))
        let cycleSum = cycle.reduce(0.0, +)
        #expect(abs(cycleSum) < 1e-7)
        
        for i in 2...13 {
            #expect(abs(result.residual[i]) < 1e-7)
        }
    }
    
    @Test("Classical additive decomposition with odd period (period = 3)")
    func testAdditiveDecompositionOddPeriod() throws {
        let series = [1.0, 4.0, 2.0, 2.0, 5.0, 3.0, 3.0, 6.0, 4.0]
        let result = try TimeSeriesDecomposition.decompose(series: series, period: 3, model: .additive)
        #expect(result.trend.count == 9)
        #expect(result.seasonal.count == 9)
        #expect(result.residual.count == 9)
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
        
        for i in 2...13 {
            #expect(abs(result.residual[i] - 1.0) < 0.02)
        }
        
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

        let infSeries = [1.0, Double.infinity, 3.0, 4.0, 5.0, 6.0]
        #expect(throws: ForecastError.self) {
            try TimeSeriesDecomposition.decompose(series: infSeries, period: 2)
        }
    }
    
    @Test("Autocorrelation function (ACF)")
    func testACF() throws {
        let series = [1.0, 2.0, 3.0, 4.0, 5.0]
        let result = try TimeSeriesDecomposition.acf(series: series, maxLag: 2)
        #expect(result.count == 3)
        #expect(result[0] == 1.0)
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
        let stationary: [Double] = [
            5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9, 5.2, 4.8, 5.0,
            5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9, 5.2, 4.8, 5.0
        ]
        let (statStat, statP) = try TimeSeriesDecomposition.adfTest(series: stationary, maxLag: 1)
        #expect(statStat < 0.0)
        #expect(statP < 0.3)

        var randomWalk = [Double](repeating: 0.0, count: 25)
        randomWalk[0] = 100.0
        var rng = SimpleRNG(seed: 42)
        for i in 1..<25 {
            randomWalk[i] = randomWalk[i-1] + rng.nextGaussian()
        }
        let (_, rwP) = try TimeSeriesDecomposition.adfTest(series: randomWalk, maxLag: 1)
        #expect(rwP > 0.05)
    }

    @Test("vDSP_convD 1D FIR moving average convolution")
    func testMovingAverageFIR() throws {
        let series = [1.0, 2.0, 3.0, 4.0, 5.0]
        let ma = TimeSeriesDecomposition.movingAverageFIR(series, period: 3)
        #expect(ma.count == 3)
        #expect(abs(ma[0] - 2.0) < 1e-7)
        #expect(abs(ma[1] - 3.0) < 1e-7)
        #expect(abs(ma[2] - 4.0) < 1e-7)

        let invalidPeriod = TimeSeriesDecomposition.movingAverageFIR(series, period: 10)
        #expect(invalidPeriod.isEmpty)

        let zeroPeriod = TimeSeriesDecomposition.movingAverageFIR(series, period: 0)
        #expect(zeroPeriod.isEmpty)
    }

    @Test("Accelerate FFT spectral decomposition (fftDecompose)")
    func testFFTDecomposition() throws {
        let series: [Double] = (0..<32).map { t in
            let trend = Double(t) * 0.1
            let seasonal = sin(2.0 * .pi * Double(t) / 8.0)
            return trend + seasonal
        }

        let result = try TimeSeriesDecomposition.fftDecompose(series: series, topKComponents: 2)
        #expect(result.trend.count == 32)
        #expect(result.seasonal.count == 32)
        #expect(result.residual.count == 32)
        #expect(result.original.count == 32)

        #expect(throws: ForecastError.self) {
            try TimeSeriesDecomposition.fftDecompose(series: [1.0, 2.0], topKComponents: 1)
        }
    }
}
