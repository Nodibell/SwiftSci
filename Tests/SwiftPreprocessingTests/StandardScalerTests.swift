import Testing
import Foundation
import SwiftDataFrame
@testable import SwiftPreprocessing

@Suite("StandardScaler Tests")
struct StandardScalerTests {
    
    @Test("StandardScaler basic transformation")
    func testStandardScalerBasic() throws {
        let scaler = StandardScaler()
        let data: [[Double]] = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]
        
        let scaled = try scaler.fitTransform(data)
        
        // Mean should be [3.0, 4.0]
        #expect(scaler.mean != nil)
        #expect(abs(scaler.mean![0] - 3.0) < 1e-9)
        #expect(abs(scaler.mean![1] - 4.0) < 1e-9)
        
        // Variance for col 0: ((1-3)^2 + (3-3)^2 + (5-3)^2)/3 = 8/3
        // Std = sqrt(8/3) = 1.632993161855452
        #expect(scaler.std != nil)
        #expect(abs(scaler.std![0] - (8.0/3.0).squareRoot()) < 1e-9)
        
        // Scaled values
        // row 0: (1 - 3)/std ≈ -1.22474487
        #expect(abs(scaled[0][0] - (-2.0 / (8.0/3.0).squareRoot())) < 1e-9)
        #expect(abs(scaled[1][0] - 0.0) < 1e-9)
        #expect(abs(scaled[2][0] - (2.0 / (8.0/3.0).squareRoot())) < 1e-9)
    }
    
    @Test("StandardScaler constant column handling")
    func testConstantColumn() throws {
        let scaler = StandardScaler()
        let data: [[Double]] = [
            [5.0, 2.0],
            [5.0, 4.0],
            [5.0, 6.0]
        ]
        
        let scaled = try scaler.fitTransform(data)
        
        // Mean of col 0 should be 5.0
        #expect(scaler.mean![0] == 5.0)
        // Std of col 0 should be 1.0 (to avoid division by zero)
        #expect(scaler.std![0] == 1.0)
        
        // All scaled values in col 0 should be 0.0
        #expect(scaled[0][0] == 0.0)
        #expect(scaled[1][0] == 0.0)
        #expect(scaled[2][0] == 0.0)
    }
    
    @Test("StandardScaler error handling")
    func testErrors() throws {
        let scaler = StandardScaler()
        
        // Transform before fit throws
        #expect(throws: PreprocessingError.self) {
            try scaler.transform([[1.0]])
        }
        
        // Empty data throws
        #expect(throws: PreprocessingError.self) {
            try scaler.fit([])
        }
        
        // Mismatched dimensions throw
        #expect(throws: PreprocessingError.self) {
            try scaler.fit([[1.0, 2.0], [1.0]])
        }
    }

    @Test("Preprocessing scaling and encoding on DataFrame")
    func testPreprocessingGlue() throws {
        let f1 = TypedColumn<Double>(name: "f1", values: [1.0, 2.0, 3.0])
        let f2 = TypedColumn<Double>(name: "f2", values: [10.0, 20.0, 30.0])
        let cat = TypedColumn<String>(name: "cat", values: ["high", "low", "high"])
        let df = try DataFrame(columns: [f1, f2, cat])

        // Standard scaling
        let (scaledStdDf, _) = try df.standardScale(columns: ["f1", "f2"])
        let scaledF1 = scaledStdDf[column: "f1", as: Double.self]?.values
        #expect(scaledF1 != nil)
        #expect(abs((scaledF1![0] ?? 0.0) + 1.2247) < 1e-3)

        // MinMax scaling
        let (scaledMinMaxDf, _) = try df.minMaxScale(columns: ["f1", "f2"])
        let minMaxF2 = scaledMinMaxDf[column: "f2", as: Double.self]?.values
        #expect(minMaxF2?.map { $0 ?? 0.0 } == [0.0, 0.5, 1.0])

        // Label Encoding
        let (encodedDf, _) = try df.labelEncode(column: "cat")
        let catEncoded = encodedDf[column: "cat", as: Int64.self]?.values
        #expect(catEncoded?.map { $0 ?? 0 } == [0, 1, 0] || catEncoded?.map { $0 ?? 0 } == [1, 0, 1])
    }
}
