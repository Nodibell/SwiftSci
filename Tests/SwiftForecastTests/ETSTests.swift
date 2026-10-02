import Testing
import Foundation
@testable import SwiftForecast

@Suite("ETSModel and PiecewiseTrendDecomposition Tests")
struct ETSTests {

    @Test("ETSModel fit and forecast")
    func testETSModelForecast() async throws {
        // Trend with seasonal fluctuations
        let series: [Double] = [10.0, 12.0, 15.0, 18.0, 22.0, 26.0, 31.0, 35.0]
        let ets = ETSModel(error: .additive, trend: .additive, seasonal: .none)

        try await ets.fit(series: series)
        let preds = try await ets.forecast(steps: 3)

        #expect(preds.count == 3)
        #expect(preds[0] > series.last!)
        #expect(preds[1] > preds[0])
    }

    @Test("ETSModel autoFit model selection based on sample-wide AICc")
    func testETSAutoFit() async throws {
        let series: [Double] = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0]
        let autoModel = try await ETSModel.autoFit(series: series, period: 1)

        let aicc = await autoModel.aicc
        #expect(aicc.isFinite && !aicc.isNaN)

        let preds = try await autoModel.forecast(steps: 2)
        #expect(preds.count == 2)
        #expect(preds[0] > 135.0)
    }

    @Test("PiecewiseTrendDecomposition linear trend forecast")
    func testPiecewiseLinearTrend() async throws {
        let series: [Double] = (0..<20).map { i in 5.0 + Double(i) * 1.5 }
        let model = PiecewiseTrendDecomposition(growth: .linear, nChangepoints: 3)

        try await model.fit(series: series)
        let preds = try await model.predict(steps: 5)

        #expect(preds.count == 5)
        #expect(preds[0] > series.last!)
        #expect(preds[4] > preds[0])
    }

    @Test("ETSModel additive vs multiplicative error produces distinct state dynamics")
    func testETSAdditiveVsMultiplicativeError() async throws {
        // Strictly positive series with seasonal component (period = 4)
        let pattern = [2.0, 5.0, 1.5, 4.0]
        let series: [Double] = (0..<16).map { i in
            10.0 + Double(i) * 2.0 + pattern[i % 4]
        }

        let modelAdd = ETSModel(error: .additive, trend: .additive, seasonal: .multiplicative, period: 4)
        let modelMult = ETSModel(error: .multiplicative, trend: .additive, seasonal: .multiplicative, period: 4)

        try await modelAdd.fit(series: series)
        try await modelMult.fit(series: series)

        let predsAdd = try await modelAdd.forecast(steps: 4)
        let predsMult = try await modelMult.forecast(steps: 4)

        #expect(predsAdd.count == 4)
        #expect(predsMult.count == 4)

        // Additive and multiplicative error state updates produce distinct forecast trajectories
        let diff = zip(predsAdd, predsMult).map { abs($0 - $1) }.reduce(0.0, +)
        #expect(diff > 1e-3)

        let llAdd = await modelAdd.logLikelihood
        let llMult = await modelMult.logLikelihood
        #expect(llAdd != llMult)
    }

    @Test("ETSModel un-fitted throws notFitted error")
    func testETSUnfittedThrowsNotFitted() async {
        let ets = ETSModel()
        await #expect(throws: ForecastError.self) {
            _ = try await ets.forecast(steps: 3)
        }
    }

    @Test("ETSModel invalid steps throws invalidHorizon")
    func testETSInvalidHorizonThrows() async throws {
        let series: [Double] = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        let ets = ETSModel()
        try await ets.fit(series: series)

        await #expect(throws: ForecastError.self) {
            _ = try await ets.forecast(steps: 0)
        }
        await #expect(throws: ForecastError.self) {
            _ = try await ets.forecast(steps: -1)
        }
    }

    @Test("ETSModel multiplicative error rejects non-positive series")
    func testETSMultiplicativeRejectsNonPositive() async {
        let nonPosSeries = [10.0, 5.0, 0.0, -2.0, 4.0, 8.0]
        let ets = ETSModel(error: .multiplicative, trend: .none, seasonal: .none)

        await #expect(throws: ForecastError.self) {
            try await ets.fit(series: nonPosSeries)
        }
    }

    @Test("ETSModel Information Criteria (AIC, AICc, BIC) calculation")
    func testETSInformationCriteria() async throws {
        let series = [10.0, 12.0, 15.0, 18.0, 22.0, 26.0, 31.0, 35.0, 40.0, 45.0]
        let ets = ETSModel(error: .additive, trend: .additive, seasonal: .none)
        try await ets.fit(series: series)

        let aic = await ets.aic
        let aicc = await ets.aicc
        let bic = await ets.bic

        #expect(aic.isFinite && !aic.isNaN)
        #expect(aicc.isFinite && !aicc.isNaN)
        #expect(bic.isFinite && !bic.isNaN)
        #expect(aicc >= aic) // AICc correction term is strictly non-negative
    }
}
