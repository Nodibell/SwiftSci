import Testing
import Foundation
@testable import SwiftForecast

@Suite("Time Series Transformers Tests")
struct TimeSeriesTransformersTests {
    
    @Test("LagTransformer generates lag feature columns")
    func testLagTransformer() {
        let series = [10.0, 20.0, 30.0, 40.0, 50.0]
        let transformer = LagTransformer(lags: [1, 2])
        let res = transformer.transform(series: series)
        
        #expect(res.features.count == 3)
        #expect(res.features[0] == [20.0, 10.0]) // at t=2 (30.0), lag1=20, lag2=10
        #expect(res.targets == [30.0, 40.0, 50.0])
    }
    
    @Test("RollingWindow calculates sliding mean and std")
    func testRollingWindow() {
        let series = [1.0, 2.0, 3.0, 4.0, 5.0]
        let window = RollingWindow(windowSize: 3)
        let res = window.transform(series: series)
        
        #expect(res.rollingMean.count == 5)
        #expect(abs(res.rollingMean[2] - 2.0) < 1e-5) // mean of [1, 2, 3] is 2
        #expect(abs(res.rollingMean[4] - 4.0) < 1e-5) // mean of [3, 4, 5] is 4
    }
}
