import Testing
import Foundation
@testable import SwiftML

@Suite("SwiftML Utilities Tests")
struct MLUtilitiesTests {
    @Test("DatasetUtilities makeClusters and makeCircles")
    func testSyntheticDatasets() {
        let clusters = DatasetUtilities.makeClusters(nSamples: 60, centers: 3)
        #expect(clusters.features.count == 60)
        #expect(clusters.labels.count == 60)

        let circles = DatasetUtilities.makeCircles(nSamples: 40)
        #expect(circles.features.count == 40)
        #expect(circles.targets.count == 40)
    }

    @Test("DataFrame-native ClassifierEstimator fit and predict")
    func testDataFrameMLBridge() async throws {
        let df = try DataFrame(columns: [
            TypedColumn<Double>(name: "x1", values: [0.0, 0.0, 1.0, 1.0]),
            TypedColumn<Double>(name: "x2", values: [0.0, 1.0, 0.0, 1.0]),
            TypedColumn<Double>(name: "y", values: [0.0, 1.0, 1.0, 0.0])
        ])

        let mlp = MLPClassifier(hiddenLayerSizes: [8], maxIter: 500, learningRate: 0.1, seed: 42)
        try await mlp.fit(df, features: ["x1", "x2"], target: "y")
        let preds = try await mlp.predict(df, features: ["x1", "x2"])
        #expect(preds.count == 4)
    }
}
