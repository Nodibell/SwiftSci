import Testing
import Foundation
@testable import SwiftML

@Suite("LinearRegression Tests")
struct LinearRegressionTests {
    
    @Test("LinearRegression convergence on clean data")
    func testConvergence() async throws {
        // Equation: y = 2 * x1 + 3 * x2 + 5
        let features: [[Double]] = [
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 3.0]
        ]
        let targets: [Double] = [10.0, 12.0, 13.0, 15.0, 20.0]
        
        let model = LinearRegression()
        try await model.fit(features: features, targets: targets, learningRate: 0.05, epochs: 1200)
        
        let w = await model.getWeights()
        let b = await model.getBias()
        
        #expect(w != nil)
        #expect(b != nil)
        
        // Check coefficients (approximate tolerance because of SGD)
        #expect(abs(w![0] - 2.0) < 0.05)
        #expect(abs(w![1] - 3.0) < 0.05)
        #expect(abs(b! - 5.0) < 0.15)
        
        // Predict
        let preds = try await model.predict(features: [[1.0, 3.0], [3.0, 1.0]])
        #expect(abs(preds[0] - 16.0) < 0.1)
        #expect(abs(preds[1] - 14.0) < 0.1)
    }
    
    @Test("LinearRegression errors")
    func testLinearRegressionErrors() async throws {
        let model = LinearRegression()
        
        // Empty input throws
        await #expect(throws: MLError.self) {
            try await model.fit(features: [], targets: [])
        }
        
        // Transform before fit throws
        await #expect(throws: MLError.self) {
            _ = try await model.predict(features: [[1.0, 2.0]])
        }
    }
}
