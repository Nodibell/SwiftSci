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

    @Test("ETSModel autoFit model selection")
    func testETSAutoFit() async throws {
        let series: [Double] = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0]
        let autoModel = try await ETSModel.autoFit(series: series, period: 1)
        
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
}
