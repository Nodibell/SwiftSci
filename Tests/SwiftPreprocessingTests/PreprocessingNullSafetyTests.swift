import Testing
import Foundation
import SwiftDataFrame
@testable import SwiftPreprocessing

@Suite("Preprocessing Null Safety Tests")
struct PreprocessingNullSafetyTests {

    @Test("withRollingMean propagates nil values correctly without forcing 0.0")
    func testRollingMeanNilPropagation() throws {
        let col = TypedColumn<Double>(name: "val", values: [10.0, nil, 30.0])
        let df = try DataFrame(columns: [col])
        
        let res = try df.withRollingMean(column: "val", window: 3)
        guard let rollingCol = res[column: "val_rolling_mean_3", as: Double.self] else {
            Issue.record("Column not found")
            return
        }
        
        #expect(rollingCol[0] == 10.0)
        #expect(rollingCol[1] == 10.0)
        // Window of 3 for index 2: [10.0, nil, 30.0] -> compactMap yields [10.0, 30.0] -> mean is 20.0
        #expect(rollingCol[2] == 20.0)
    }

    @Test("withRollingStd ignores nil values and computes std on valid elements")
    func testRollingStdNilPropagation() throws {
        let col = TypedColumn<Double>(name: "val", values: [10.0, nil, 30.0])
        let df = try DataFrame(columns: [col])
        
        let res = try df.withRollingStd(column: "val", window: 3)
        guard let rollingCol = res[column: "val_rolling_std_3", as: Double.self] else {
            Issue.record("Column not found")
            return
        }
        
        #expect(rollingCol[0] == 0.0)
        #expect(rollingCol[1] == 0.0)
        // Window of 3 for index 2: valid elements [10.0, 30.0], std = sqrt(((10-20)^2 + (30-20)^2) / 1) = sqrt(200)
        let expectedStd = sqrt(200.0)
        let actualStd = rollingCol[2] ?? 0.0
        #expect(abs(actualStd - expectedStd) < 1e-5)
    }

    @Test("withEWMA throws DataFrameError on invalid alpha and handles nil values")
    func testEWMAInvalidAlphaAndNil() throws {
        let col = TypedColumn<Double>(name: "val", values: [10.0, nil, 20.0])
        let df = try DataFrame(columns: [col])
        
        #expect(throws: DataFrameError.self) {
            _ = try df.withEWMA(column: "val", alpha: 0.0)
        }
        
        #expect(throws: DataFrameError.self) {
            _ = try df.withEWMA(column: "val", alpha: 1.5)
        }
        
        let res = try df.withEWMA(column: "val", alpha: 0.5)
        guard let ewmaCol = res[column: "val_ewma_0.50", as: Double.self] else {
            Issue.record("Column not found")
            return
        }
        
        #expect(ewmaCol[0] == 10.0)
        #expect(ewmaCol[1] == nil)
        // Row 2: alpha * 20.0 + (1 - alpha) * 10.0 = 15.0
        #expect(ewmaCol[2] == 15.0)
    }
}
