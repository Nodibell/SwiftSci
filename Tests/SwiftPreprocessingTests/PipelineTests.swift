import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("Pipeline Tests")
struct PipelineTests {
    
    @Test("Pipeline chains Imputer and StandardScaler")
    func testPipelineChaining() throws {
        let imputer = Imputer(strategy: .mean)
        var scaler = StandardScaler()
        var pipeline = Pipeline(steps: [imputer, scaler])
        
        let trainData = [
            [2.0],
            [Double.nan],
            [4.0]
        ]
        
        // Imputation should fill NaN with mean (3.0).
        // Imputed: [2.0], [3.0], [4.0]
        // Mean of imputed is 3.0. Stddev of imputed is sqrt(2/3) ≈ 0.816496580927726
        
        let transformed = try pipeline.fitTransform(trainData)
        
        #expect(abs(transformed[0][0] - (-1.224744871391589)) < 1e-6)
        #expect(abs(transformed[1][0] - 0.0) < 1e-6)
        #expect(abs(transformed[2][0] - 1.224744871391589) < 1e-6)
    }

    @Test("Pipeline fit followed by separate transform on new data")
    func testPipelineFitThenSeparateTransform() throws {
        let scaler = MinMaxScaler()
        let pipeline = Pipeline(steps: [scaler])
        
        let trainData = [[10.0], [20.0]]
        let testData = [[15.0]]
        
        try pipeline.fit(trainData)
        let transformedTest = try pipeline.transform(testData)
        
        // Min = 10, Max = 20 -> (15 - 10)/(20 - 10) = 0.5
        #expect(abs(transformedTest[0][0] - 0.5) < 1e-6)
    }
}
