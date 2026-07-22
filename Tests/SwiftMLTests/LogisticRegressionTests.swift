import Testing
import Foundation
@testable import SwiftML

@Suite("LogisticRegression Tests")
struct LogisticRegressionTests {
    
    @Test("LogisticRegression classification of linearly separable data")
    func testLogisticClassification() async throws {
        // Linearly separable dataset: feature > 0 -> class 1, else class 0
        let features: [[Double]] = [[-2.0], [-1.0], [1.0], [2.0]]
        let targets: [Double] = [0.0, 0.0, 1.0, 1.0]
        
        let model = LogisticRegression()
        try await model.fit(features: features, targets: targets, learningRate: 0.5, epochs: 600)
        
        let w = await model.getWeights()
        let b = await model.getBias()
        
        #expect(w != nil)
        #expect(b != nil)
        
        let testFeatures: [[Double]] = [[-3.0], [3.0]]
        let probs = try await model.predictProbability(features: testFeatures)
        #expect(probs[0][1] < 0.2) // Prob of class 1 for -3.0 should be very small
        #expect(probs[1][1] > 0.8) // Prob of class 1 for 3.0 should be very high
        
        let classes = try await model.predict(features: testFeatures)
        #expect(classes[0] == 0)
        #expect(classes[1] == 1)
    }
    
    @Test("LogisticRegression errors")
    func testLogisticRegressionErrors() async throws {
        let model = LogisticRegression()
        
        // Empty input throws
        await #expect(throws: MLError.self) {
            try await model.fit(features: [], targets: [])
        }
        
        // Transform before fit throws
        await #expect(throws: MLError.self) {
            _ = try await model.predict(features: [[1.0]])
        }
    }
}
