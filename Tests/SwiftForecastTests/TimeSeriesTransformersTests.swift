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
    func testRollingWindow() throws {
        let series = [1.0, 2.0, 3.0, 4.0, 5.0]
        let window = RollingWindow(windowSize: 3)
        let res = try window.transform(series: series)

        
        #expect(res.rollingMean.count == 5)
        #expect(abs(res.rollingMean[2] - 2.0) < 1e-5) // mean of [1, 2, 3] is 2
        #expect(abs(res.rollingMean[4] - 4.0) < 1e-5) // mean of [3, 4, 5] is 4
    }

    @Test("RollingWindow O(N) numerical stability on large offset values (M-03)")
    func testRollingWindowNumericalStability() throws {
        // Series with large constant baseline: 1e8 + [1, 2, 3, 4, 5]
        let base = 1e8
        let series = [base + 1.0, base + 2.0, base + 3.0, base + 4.0, base + 5.0]
        let window = RollingWindow(windowSize: 3)
        let res = try window.transform(series: series)

        #expect(res.rollingMean.count == 5)
        #expect(abs(res.rollingMean[2] - (base + 2.0)) < 1e-7)
        #expect(abs(res.rollingMean[4] - (base + 4.0)) < 1e-7)

        // Std of [1, 2, 3] is 1.0
        #expect(abs(res.rollingStd[2] - 1.0) < 1e-6)
        #expect(abs(res.rollingStd[4] - 1.0) < 1e-6)
    }

    @Test("ExpandingWindow Welford algorithm on high-magnitude series (M-04)")
    func testExpandingWindowWelford() {
        let base = 1e9
        let series = [base + 2.0, base + 4.0, base + 6.0]
        let expanding = ExpandingWindow(minPeriods: 2)
        let res = expanding.transform(series: series)

        #expect(res.expandingMean[0].isNaN)
        #expect(res.expandingStd[0].isNaN)

        // Mean of [2, 4] is 3.0, std is sqrt(2) ≈ 1.41421356
        #expect(abs(res.expandingMean[1] - (base + 3.0)) < 1e-7)
        #expect(abs(res.expandingStd[1] - sqrt(2.0)) < 1e-6)

        // Mean of [2, 4, 6] is 4.0, std is 2.0
        #expect(abs(res.expandingMean[2] - (base + 4.0)) < 1e-7)
        #expect(abs(res.expandingStd[2] - 2.0) < 1e-6)
    }

    @Test("RollingWindow high-magnitude baseline (1e12) with fluctuations avoids variance cancellation")
    func testRollingWindowHighMagnitudeFluctuations() throws {
        let base = 1e12
        let n = 200
        let series = (0..<n).map { base + Double($0 % 5) }
        let window = RollingWindow(windowSize: 10)
        let res = try window.transform(series: series)

        #expect(res.rollingMean.count == n)
        #expect(res.rollingStd.count == n)
        for i in 10..<n {
            #expect(!res.rollingMean[i].isNaN && res.rollingMean[i].isFinite)
            #expect(!res.rollingStd[i].isNaN && res.rollingStd[i] >= 0.0)
        }
    }
}
