import Testing
import SwiftML

@Suite("Legacy regression failure compatibility")
struct LegacyRegressionErrorTests {
    @Test(arguments: [Double.nan, Double.infinity, -Double.infinity])
    func nonfiniteTrainingRejectsNonfiniteValues(value: Double) async throws {
        for (features, targets) in [([[1.0]], [value]), ([[value]], [1.0])] {
            let model = LinearRegression(device: .cpu)
            await #expect(throws: SwiftMLError.self) { try await model.fit(features: features, targets: targets) }
            await #expect(throws: SwiftMLError.self) {
                try await model.fitCPUGradientDescent(features: features, targets: targets)
            }
        }
    }

    @Test func zeroEpochGradientDescentRejectsNonPositiveEpochs() async throws {
        let model = LinearRegression(device: .cpu)
        await #expect(throws: SwiftMLError.self) {
            try await model.fitCPUGradientDescent(features: [[1.0]], targets: [1.0], epochs: 0)
        }
    }
}
