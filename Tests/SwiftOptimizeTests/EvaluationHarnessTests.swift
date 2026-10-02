import Testing
import Foundation
@testable import SwiftOptimize
import SwiftML

@Suite("Evaluation Harness Tests (G-019)")
struct EvaluationHarnessTests {

    @Test("evaluateClassification computes complete metrics suite and confusion matrix")
    func testClassificationEvaluation() throws {
        let yTrue = [0, 0, 1, 1, 2, 2]
        let yPred = [0, 1, 1, 1, 2, 0]

        let result = try EvaluationHarness.evaluateClassification(yTrue: yTrue, yPred: yPred)

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
    func testStringClassificationEvaluation() throws {
        let yTrue = ["cat", "cat", "dog", "dog"]
        let yPred = ["cat", "dog", "dog", "dog"]

        let (result, mapping) = try EvaluationHarness.evaluateClassification(yTrue: yTrue, yPred: yPred)
        #expect(mapping.count == 2)
        #expect(result.accuracy == 0.75)
        #expect(result.confusionMatrix.count == 2)
    }

    @Test("evaluateClassification with integer-valued Double labels succeeds")
    func testDoubleClassificationEvaluation() throws {
        let yTrue = [0.0, 1.0, 2.0, 1.0]
        let yPred = [0.0, 1.0, 1.0, 1.0]

        let result = try EvaluationHarness.evaluateClassification(yTrue: yTrue, yPred: yPred)
        #expect(result.accuracy == 0.75)
    }

    @Test("evaluateClassification rejects non-integer Double labels with EvaluationError")
    func testDoubleClassificationRejectsContinuousValues() {
        let yTrue = [0.2, 0.8, 1.5]
        let yPred = [0.0, 1.0, 1.0]

        #expect(throws: EvaluationError.self) {
            _ = try EvaluationHarness.evaluateClassification(yTrue: yTrue, yPred: yPred)
        }
    }

    @Test("evaluateClassification rejects empty data and dimension mismatch")
    func testClassificationValidationErrors() {
        // Empty data
        #expect(throws: EvaluationError.self) {
            _ = try EvaluationHarness.evaluateClassification(yTrue: [Int](), yPred: [Int]())
        }

        // Mismatched dimensions
        #expect(throws: EvaluationError.self) {
            _ = try EvaluationHarness.evaluateClassification(yTrue: [0, 1], yPred: [0])
        }
    }

    @Test("evaluateRegression computes R^2, MSE, RMSE, MAE, and MAPE")
    func testRegressionEvaluation() throws {
        let yTrue = [10.0, 20.0, 30.0, 40.0, 50.0]
        let yPred = [11.0, 19.0, 32.0, 39.0, 51.0]

        let result = try EvaluationHarness.evaluateRegression(yTrue: yTrue, yPred: yPred, numFeatures: 2)

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

    @Test("evaluateRegression rejects empty data and dimension mismatch")
    func testRegressionValidationErrors() {
        #expect(throws: EvaluationError.self) {
            _ = try EvaluationHarness.evaluateRegression(yTrue: [], yPred: [])
        }
        #expect(throws: EvaluationError.self) {
            _ = try EvaluationHarness.evaluateRegression(yTrue: [1.0, 2.0], yPred: [1.0])
        }
    }

    @Test("evaluateRegression rejects invalid numFeatures parameters")
    func testRegressionNumFeaturesValidation() {
        // n = 4, p = 3 -> n not > p + 1 (4 not > 4) -> throws
        #expect(throws: EvaluationError.self) {
            _ = try EvaluationHarness.evaluateRegression(yTrue: [1.0, 2.0, 3.0, 4.0], yPred: [1.1, 1.9, 3.1, 3.9], numFeatures: 3)
        }
        // p <= 0 -> throws
        #expect(throws: EvaluationError.self) {
            _ = try EvaluationHarness.evaluateRegression(yTrue: [1.0, 2.0, 3.0, 4.0], yPred: [1.1, 1.9, 3.1, 3.9], numFeatures: 0)
        }
        // n = 4, p = 1 -> n > p + 1 (4 > 2) -> valid
        #expect(throws: Never.self) {
            _ = try EvaluationHarness.evaluateRegression(yTrue: [1.0, 2.0, 3.0, 4.0], yPred: [1.1, 1.9, 3.1, 3.9], numFeatures: 1)
        }
    }
}
