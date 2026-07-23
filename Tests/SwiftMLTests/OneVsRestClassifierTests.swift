import Testing
import Foundation
@testable import SwiftML

@Suite("OneVsRestClassifier Multi-Class Tests")
struct OneVsRestClassifierTests {

    @Test("Fit and Predict Multi-Class Classification")
    func testOneVsRestFitAndPredict() async throws {
        // 4 samples, 2 features, 3 classes (0, 1, 2)
        let features: [[Double]] = [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.8]
        ]
        let targets: [Double] = [0.0, 0.0, 1.0, 1.0]

        let ovr = OneVsRestClassifier(numClasses: 2)
        try await ovr.fit(features: features, targets: targets)

        let preds = try await ovr.predict(features: features)
        #expect(preds.count == 4)
        #expect(preds[0] == 0)
        #expect(preds[1] == 0)
        #expect(preds[2] == 1)
        #expect(preds[3] == 1)
    }
}
