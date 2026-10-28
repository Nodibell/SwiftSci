import Testing
import Foundation
@testable import SwiftForecast

@Suite("Weighted Absolute Percentage Error (WAPE) Tests (G-029)")
struct WAPETests {

    @Test("Analytical WAPE numerical parity")
    func testAnalyticalParity() throws {
        let actual = [100.0, 150.0, 200.0]
        let forecast = [110.0, 140.0, 210.0]
        // Sum of absolute errors = 10 + 10 + 10 = 30
        // Sum of actuals = 100 + 150 + 200 = 450
        // WAPE = (30 / 450) * 100% = 6.666666666666667%
        let wapeVal = TimeSeriesMetrics.wape(actual: actual, forecast: forecast)
        #expect(abs(wapeVal - (30.0 / 450.0 * 100.0)) < 1e-10)

        let evaluated = try TimeSeriesMetrics.evaluateWAPE(actual: actual, forecast: forecast)
        #expect(abs(evaluated - wapeVal) < 1e-12)
    }

    @Test("Perfect forecast yields 0% WAPE")
    func testPerfectForecast() throws {
        let actual = [12.5, 45.0, 89.2, 104.7]
        let forecast = [12.5, 45.0, 89.2, 104.7]
        let val = TimeSeriesMetrics.wape(actual: actual, forecast: forecast)
        #expect(val == 0.0)
        #expect(try TimeSeriesMetrics.evaluateWAPE(actual: actual, forecast: forecast) == 0.0)
    }

    @Test("Negative values are properly absolute-summed")
    func testNegativeValues() throws {
        let actual = [-50.0, 50.0]
        let forecast = [-40.0, 60.0]
        // Sum of errors: |-50 - (-40)| + |50 - 60| = 10 + 10 = 20
        // Sum of actuals: |-50| + |50| = 100
        // WAPE = 20 / 100 * 100% = 20%
        let val = TimeSeriesMetrics.wape(actual: actual, forecast: forecast)
        #expect(abs(val - 20.0) < 1e-10)
    }

    @Test("Zero-sum actual series edge case handling")
    func testZeroSumActuals() throws {
        // Both zeros -> 0.0%
        let zeros = [0.0, 0.0, 0.0]
        #expect(TimeSeriesMetrics.wape(actual: zeros, forecast: zeros) == 0.0)

        // Non-zero forecast for zero actuals -> 100.0% penalty in safe method
        let nonZeroForecast = [1.0, 2.0, 3.0]
        #expect(TimeSeriesMetrics.wape(actual: zeros, forecast: nonZeroForecast) == 100.0)
    }

    @Test("SIMD Accelerate vs scalar reduction parity across 100 points")
    func testVectorizedParity() throws {
        var actual = [Double]()
        var forecast = [Double]()
        for i in 1...100 {
            let act = Double(i) * 1.5 + sin(Double(i)) * 5.0
            let pred = act + cos(Double(i)) * 2.0
            actual.append(act)
            forecast.append(pred)
        }

        let simdWape = TimeSeriesMetrics.wape(actual: actual, forecast: forecast)

        // Scalar implementation
        let totalAbsDiff = zip(actual, forecast).reduce(0.0) { $0 + abs($1.0 - $1.1) }
        let totalActual = actual.reduce(0.0) { $0 + abs($1) }
        let scalarWape = (totalAbsDiff / totalActual) * 100.0

        #expect(abs(simdWape - scalarWape) < 1e-10)
    }

    @Test("evaluateWAPE throws on invalid inputs")
    func testThrowingValidation() {
        // Empty series
        #expect(throws: ForecastError.self) {
            try TimeSeriesMetrics.evaluateWAPE(actual: [], forecast: [])
        }

        // Dimension mismatch
        #expect(throws: ForecastError.self) {
            try TimeSeriesMetrics.evaluateWAPE(actual: [1.0, 2.0], forecast: [1.0])
        }

        // NaN
        #expect(throws: ForecastError.self) {
            try TimeSeriesMetrics.evaluateWAPE(actual: [Double.nan, 2.0], forecast: [1.0, 2.0])
        }

        // Infinity
        #expect(throws: ForecastError.self) {
            try TimeSeriesMetrics.evaluateWAPE(actual: [1.0, 2.0], forecast: [Double.infinity, 2.0])
        }

        // Invalid epsilon (negative, NaN, infinity) across evaluateMAPE, evaluateSMAPE, evaluateWAPE
        for badEpsilon in [-1.0, -1e-9, Double.nan, Double.infinity, -Double.infinity] {
            #expect(throws: ForecastError.self) {
                try TimeSeriesMetrics.evaluateMAPE(actual: [0.0, 2.0], forecast: [1.0, 2.0], epsilon: badEpsilon)
            }
            #expect(throws: ForecastError.self) {
                try TimeSeriesMetrics.evaluateSMAPE(actual: [0.0, 2.0], forecast: [0.0, 2.0], epsilon: badEpsilon)
            }
            #expect(throws: ForecastError.self) {
                try TimeSeriesMetrics.evaluateWAPE(actual: [0.0, 2.0], forecast: [1.0, 2.0], epsilon: badEpsilon)
            }
        }
    }

    @Test("Backtester result includes WAPE in fold and summary metrics")
    func testBacktestFoldAndResultWAPE() async throws {
        let series = [10.0, 12.0, 15.0, 14.0, 18.0, 21.0, 25.0, 24.0, 28.0, 32.0]
        let result = try await TimeSeriesBacktester.backtest(
            series: series,
            initialWindow: 5,
            horizon: 1,
            stepSize: 1,
            mode: .expanding
        ) { train, h in
            [train.last!] // 1-step persistence baseline
        }

        #expect(!result.folds.isEmpty)
        for fold in result.folds {
            #expect(fold.wape >= 0.0)
        }
        #expect(result.wape >= 0.0)
        #expect(result.metrics["wape"] == result.wape)
    }
}
