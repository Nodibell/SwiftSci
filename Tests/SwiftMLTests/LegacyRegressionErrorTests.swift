import Testing
import SwiftML

@Suite("Legacy regression failure compatibility")
struct LegacyRegressionErrorTests {
    @Test(arguments: [Double.nan, Double.infinity, -Double.infinity])
    func nonfiniteTrainingPreservesDivergenceError(value: Double) async throws {
        let expected = SwiftMLError.trainingFailed(
            "Gradient descent diverged: weights or bias contains NaN or Infinity. Try a lower learning rate."
        )
        for (features, targets) in [([[1.0]], [value]), ([[value]], [1.0])] {
            let model = LinearRegression(device: .cpu)
            await #expect(throws: expected) { try await model.fit(features: features, targets: targets) }
            await #expect(throws: expected) {
                try await model.fitCPUGradientDescent(features: features, targets: targets)
            }
        }
    }

    @Test func zeroEpochGradientDescentRetainsLegacyInitialization() async throws {
        let model = LinearRegression(device: .cpu)
        try await model.fitCPUGradientDescent(features: [[.nan]], targets: [.nan], epochs: 0)
        let state = await model.getWeightsAndBias()
        #expect(state.weights == [0])
        #expect(state.bias == 0)
    }
}
