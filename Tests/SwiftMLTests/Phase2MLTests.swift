import Testing
import Foundation
@testable import SwiftML

@Suite("Phase 2 ML & Calibration Tests")
struct Phase2MLTests {
    
    @Test("CalibratedClassifier calibrates base decision tree probabilities")
    func testCalibratedClassifier() async throws {
        let dataset = DatasetUtilities.makeClassification(nSamples: 60, nFeatures: 2, seed: 42)
        
        let dt = DecisionTreeClassifier(maxDepth: 3)
        let calibrated = CalibratedClassifier(baseEstimator: dt)
        
        try await calibrated.fit(features: dataset.features, targets: dataset.targets)
        
        let probs = try await calibrated.predictProbability(features: dataset.features)
        #expect(probs.count == 60)
        #expect(probs[0].count == 2)
        #expect(abs(probs[0][0] + probs[0][1] - 1.0) < 1e-5)
    }
    
    @Test("DatasetUtilities generates synthetic datasets")
    func testDatasetUtilities() {
        let clf = DatasetUtilities.makeClassification(nSamples: 40, nFeatures: 3, nClasses: 2, seed: 123)
        #expect(clf.features.count == 40)
        #expect(clf.features[0].count == 3)
        #expect(clf.targets.count == 40)
        
        let reg = DatasetUtilities.makeRegression(nSamples: 30, nFeatures: 4, seed: 456)
        #expect(reg.features.count == 30)
        #expect(reg.features[0].count == 4)
        #expect(reg.targets.count == 30)
    }
}
