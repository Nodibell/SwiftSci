import Testing
import SwiftML
import SwiftPreprocessing
import SwiftDataFrame

@Suite("Public standardized MLP failed-fit state")
struct StandardizedMLPFailedFitTests {
    @Test func divergentFitPreservesPreviouslyCommittedWeights() async throws {
        let training = try publicMLPTraining(), model = publicMLPModel()
        _ = try await model.fitStandardized(features: training.batch, targets: training.targets)
        var pipeline = Pipeline(steps: [Imputer(), StandardScaler()])
        try pipeline.fit(training.batch)
        let clean = try pipeline.transform(training.batch)
        let before = try await model.predict(features: clean)
        let divergent = [Double](repeating: .greatestFiniteMagnitude, count: training.batch.rowCount)
        await #expect(throws: SwiftMLError.self) {
            try await model.fitStandardized(features: training.batch, targets: divergent)
        }
        let after = try await model.predict(features: clean)
        #expect(after.map(\.bitPattern) == before.map(\.bitPattern))
    }
    @Test func invalidConfigurationThrowsBeforeTraining() async throws {
        let training = try publicMLPTraining()
        let model = MLPRegressor(hiddenLayerSizes: [0], maxIter: 1, requestedDevice: .cpu)
        await #expect(throws: SwiftMLError.self) {
            try await model.fitStandardized(features: training.batch, targets: training.targets)
        }
    }
}
