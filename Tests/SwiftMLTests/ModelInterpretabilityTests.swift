import Testing
import Foundation
@testable import SwiftML

@Suite("Model Interpretability & Feature Importance Tests")
struct ModelInterpretabilityTests {
    
    @Test("OneVsRest topFeatures with featureNames and vocabulary")
    func testOneVsRestTopFeatures() async throws {
        // 3 classes, 3 features:
        // class 0 strongly correlated with feature 0 ("sport")
        // class 1 strongly correlated with feature 1 ("politics")
        // class 2 strongly correlated with feature 2 ("economy")
        let features: [[Double]] = [
            [5.0, 0.0, 0.0],
            [4.5, 0.1, 0.0],
            [0.0, 5.0, 0.0],
            [0.1, 4.8, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.2, 4.7]
        ]
        let targets: [Double] = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        
        let ovr = OneVsRestClassifier(numClasses: 3)
        try await ovr.fit(features: features, targets: targets, epochs: 500)
        
        let vocab = ["sport": 0, "politics": 1, "economy": 2]
        
        let topClass0 = try await ovr.topFeatures(classIndex: 0, topN: 1, vocabulary: vocab)
        #expect(topClass0.count == 1)
        #expect(topClass0[0].feature == "sport")
        #expect(topClass0[0].weight > 0.0)
        
        let topClass1 = try await ovr.topFeatures(classIndex: 1, topN: 1, vocabulary: vocab)
        #expect(topClass1.count == 1)
        #expect(topClass1[0].feature == "politics")
        
        let topClass2 = try await ovr.topFeatures(classIndex: 2, topN: 1, vocabulary: vocab)
        #expect(topClass2.count == 1)
        #expect(topClass2[0].feature == "economy")
        
        let allPerClass = try await ovr.topFeaturesPerClass(topN: 2, vocabulary: vocab)
        #expect(allPerClass.count == 3)
        #expect(allPerClass[0]?.count == 2)
    }
    
    @Test("LinearSVCOneVsRest topFeatures")
    func testLinearSVCOneVsRestTopFeatures() async throws {
        let features: [[Double]] = [
            [10.0, 0.0],
            [9.0, 0.2],
            [0.0, 10.0],
            [0.1, 9.5]
        ]
        let targets: [Double] = [0.0, 0.0, 1.0, 1.0]
        
        let svcOvr = LinearSVCOneVsRest(numClasses: 2)
        try await svcOvr.fit(features: features, targets: targets, epochs: 300)
        
        let names = ["alpha", "beta"]
        let top0 = try await svcOvr.topFeatures(classIndex: 0, topN: 2, featureNames: names)
        #expect(top0.count == 2)
        #expect(top0[0].feature == "alpha")
        #expect(top0[0].weight > top0[1].weight)
        
        let top1 = try await svcOvr.topFeatures(classIndex: 1, topN: 2, featureNames: names)
        #expect(top1.count == 2)
        #expect(top1[0].feature == "beta")
        #expect(top1[0].weight > top1[1].weight)
    }
}
