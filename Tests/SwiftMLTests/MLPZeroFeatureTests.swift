import Testing
@testable import SwiftML

@Suite("MLP zero-feature input")
struct MLPZeroFeatureTests {
    @Test func regressorRejectsZeroFeatureTraining() async throws {
        let model = MLPRegressor(hiddenLayerSizes: [2], maxIter: 0, requestedDevice: .cpu)
        let features: [[Double]] = [[], []]
        await #expect(throws: SwiftMLError.emptyInput) {
            try await model.fit(features: features, targets: [1, 2])
        }
        #expect(await model.trainedLayers == nil)
    }

    @Test func classifierRejectsZeroFeatureTraining() async throws {
        let model = MLPClassifier(hiddenLayerSizes: [2], maxIter: 0, requestedDevice: .cpu)
        let features: [[Double]] = [[], []]
        await #expect(throws: SwiftMLError.emptyInput) {
            try await model.fit(features: features, targets: [0, 1])
        }
        #expect(await model.trainedLayers == nil)
    }

    @Test(arguments: [32, 33])
    func batchedPredictionRejectsZeroInputDimension(rows: Int) throws {
        let layer = LayerWeights(W: [], b: [0.5], inDim: 0, outDim: 1)
        let features = [[Double]](repeating: [], count: rows)
        #expect(throws: SwiftMLError.emptyInput) {
            let _: [Double]? = try mlpBatchedPrediction(
                features: features, layers: [layer],
                activate: { $0 }, finish: { $0[0] })
        }
    }
}
