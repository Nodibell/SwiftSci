import Foundation
import MLX
import Testing
import SwiftPreprocessing
@testable import SwiftML

@Suite("Binary cross entropy gradients", .serialized)
struct BinaryCrossEntropyTests {
    @Test("Zero-logit derivative is one half minus the target", arguments: [Float(0), 0.25, 1])
    func zeroLogitGradient(target: Float) {
        let differentiate = valueAndGrad({ (arguments: [MLXArray]) -> [MLXArray] in
            [Losses.binaryCrossEntropy(logits: arguments[0], targets: MLXArray(target))]
        }, argumentNumbers: [0])
        let (loss, gradients) = differentiate([MLXArray(Float(0))])
        #expect(abs(loss[0].item(Float.self) - Float(log(2.0))) < 1e-6)
        #expect(abs(gradients[0].item(Float.self) - (0.5 - target)) < 1e-6)
    }

    @Test("Mean loss has finite, analytical gradients for mixed and extreme logits", arguments: [Float(0), 0.25, 1])
    func meanGradient(target: Float) {
        let logits: [Float] = [-1_000, -2, 0, 2, 1_000]
        let probabilities: [Double] = [0, 0.11920292202211755, 0.5, 0.8807970779778823, 1]
        let differentiate = valueAndGrad({ (arguments: [MLXArray]) -> [MLXArray] in
            [Losses.binaryCrossEntropy(logits: arguments[0], targets: MLXArray(Array(repeating: target, count: 5)).reshaped([5, 1]))]
        }, argumentNumbers: [0])
        let (loss, gradients) = differentiate([MLXArray(logits).reshaped([5, 1])])
        #expect(loss[0].item(Float.self).isFinite)
        let actual = gradients[0].asArray(Float.self)
        for index in logits.indices {
            let expected = Float((probabilities[index] - Double(target)) / Double(logits.count))
            #expect(actual[index].isFinite)
            #expect(abs(actual[index] - expected) < 1e-6)
        }
    }

    @Test("One GPU training step updates a zero-logit bias", arguments: [Double(0), 0.25, 1])
    func firstTrainingStep(target: Double) async throws {
        let model = LogisticRegression(device: .gpu)
        try await model.fit(features: [[0.0]], targets: [target], learningRate: 0.25, epochs: 1)
        let parameters = await model.getWeightsAndBias()
        #expect(await model.resolvedDevice == .gpu)
        #expect(parameters.weights == [0.0])
        let bias = try #require(parameters.bias)
        #expect(abs(bias - (-0.25 * (0.5 - target))) < 1e-6)
    }
}
