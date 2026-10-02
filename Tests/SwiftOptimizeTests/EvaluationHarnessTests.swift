import Testing
import Foundation
@testable import SwiftOptimize
import SwiftML

@Suite("Evaluation Harness Tests (G-019)")
struct EvaluationHarnessTests {

    @Test("evaluateClassification computes complete metrics suite and confusion matrix")
    func testClassificationEvaluation() {
        let yTrue = [0, 0, 1, 1, 2, 2]
        let yPred = [0, 1, 1, 1, 2, 0]

        let result = EvaluationHarness.evaluateClassification(yTrue: yTrue, yPred: yPred)

        #expect(result.classLabels == [0, 1, 2])
        #expect(result.confusionMatrix.count == 3)
        #expect(result.confusionMatrix[0] == [1, 1, 0]) // true 0: 1 pred 0, 1 pred 1
        #expect(result.confusionMatrix[1] == [0, 2, 0]) // true 1: 0 pred 0, 2 pred 1
        #expect(result.confusionMatrix[2] == [1, 0, 1]) // true 2: 1 pred 0, 1 pred 2

        #expect(result.accuracy == 4.0 / 6.0)
        #expect(result.macroPrecision > 0.0)
        #expect(result.macroRecall > 0.0)
        #expect(result.macroF1 > 0.0)
        #expect(result.weightedF1 > 0.0)

        let report = result.toEvaluationReport()
        #expect(report.confusionMatrix == result.confusionMatrix)
        #expect(report.metrics["accuracy"] == result.accuracy)
    }

    @Test("evaluateClassification with String labels")
    func testStringClassificationEvaluation() {
        let yTrue = ["cat", "cat", "dog", "dog"]
        let yPred = ["cat", "dog", "dog", "dog"]

        let (result, mapping) = EvaluationHarness.evaluateClassification(yTrue: yTrue, yPred: yPred)
        #expect(mapping.count == 2)
        #expect(result.accuracy == 0.75)
        #expect(result.confusionMatrix.count == 2)
    }

    @Test("evaluateRegression computes R^2, MSE, RMSE, MAE, and MAPE")
    func testRegressionEvaluation() {
        let yTrue = [10.0, 20.0, 30.0, 40.0, 50.0]
        let yPred = [11.0, 19.0, 32.0, 39.0, 51.0]

        let result = EvaluationHarness.evaluateRegression(yTrue: yTrue, yPred: yPred, numFeatures: 2)

        #expect(result.r2 > 0.95)
        #expect(result.adjustedR2 != nil)
        #expect(result.mse > 0.0)
        #expect(result.rmse == sqrt(result.mse))
        #expect(result.mae > 0.0)
        #expect(result.mape > 0.0)
        #expect(result.metrics["r2"] == result.r2)

        let report = result.toEvaluationReport()
        #expect(report.metrics["r2"] == result.r2)
        #expect(report.confusionMatrix == nil)
    }
}
