import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("Phase 2 Preprocessing Tests")
struct Phase2PreprocessingTests {
    
    @Test("SMOTE generates synthetic minority samples")
    func testSMOTEOversampling() throws {
        // Imbalanced dataset: 10 majority class 0, 3 minority class 1
        var features: [[Double]] = []
        var targets: [Double] = []
        
        for _ in 0..<10 {
            features.append([Double.random(in: 0.0...1.0), Double.random(in: 0.0...1.0)])
            targets.append(0.0)
        }
        for _ in 0..<3 {
            features.append([Double.random(in: 10.0...11.0), Double.random(in: 10.0...11.0)])
            targets.append(1.0)
        }
        
        let smote = SMOTE(kNeighbors: 2, seed: 42)
        let resampled = try smote.fitResample(features: features, targets: targets)
        
        let class0Count = resampled.targets.filter { $0 == 0.0 }.count
        let class1Count = resampled.targets.filter { $0 == 1.0 }.count
        
        #expect(class0Count == 10)
        #expect(class1Count == 10)
    }
    
    @Test("VarianceThreshold drops constant features")
    func testVarianceThreshold() throws {
        // Feature 0 has zero variance (constant 5.0), Feature 1 varies
        let data: [[Double]] = [[5.0, 1.0], [5.0, 2.0], [5.0, 3.0], [5.0, 4.0]]
        let selector = VarianceThreshold(threshold: 0.0)
        try selector.fit(data)
        let transformed = try selector.transform(data)
        
        #expect(transformed[0].count == 1)
        #expect(transformed[0][0] == 1.0)
    }
    
    @Test("InteractionFeatures generates pairwise feature products")
    func testInteractionFeatures() throws {
        let data: [[Double]] = [[2.0, 3.0]]
        let interaction = InteractionFeatures()
        let transformed = try interaction.transform(data)
        
        #expect(transformed[0] == [2.0, 3.0, 6.0])
    }
}
