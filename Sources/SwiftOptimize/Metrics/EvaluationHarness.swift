import Foundation
import SwiftML

/// Standardized evaluation metrics result for classification models (G-019).
public struct ClassificationEvaluation: Sendable, Codable, Equatable {
    /// Overall classification accuracy in [0, 1].
    public let accuracy: Double
    /// Unweighted macro-averaged precision across all observed classes.
    public let macroPrecision: Double
    /// Unweighted macro-averaged recall across all observed classes.
    public let macroRecall: Double
    /// Unweighted macro-averaged F1 score across all observed classes.
    public let macroF1: Double
    /// Weighted F1 score proportional to class support counts.
    public let weightedF1: Double
    /// Sorted unique class label identifiers.
    public let classLabels: [Int]
    /// K x K confusion matrix where rows are ground-truth labels and columns are predictions.
    public let confusionMatrix: [[Int]]
    /// Flat dictionary of key evaluation metrics for tabular leaderboards and logging.
    public let metrics: [String: Double]

    /// Creates a new ClassificationEvaluation instance.
    public init(
        accuracy: Double,
        macroPrecision: Double,
        macroRecall: Double,
        macroF1: Double,
        weightedF1: Double,
        classLabels: [Int],
        confusionMatrix: [[Int]],
        metrics: [String: Double]? = nil
    ) {
        self.accuracy = accuracy
        self.macroPrecision = macroPrecision
        self.macroRecall = macroRecall
        self.macroF1 = macroF1
        self.weightedF1 = weightedF1
        self.classLabels = classLabels
        self.confusionMatrix = confusionMatrix
        if let m = metrics {
            self.metrics = m
        } else {
            self.metrics = [
                "accuracy": accuracy,
                "precision": macroPrecision,
                "recall": macroRecall,
                "f1": macroF1,
                "macroF1": macroF1,
                "weightedF1": weightedF1
            ]
        }
    }

    /// Converts this result into a standardized `EvaluationReport` from `SwiftML`.
    public func toEvaluationReport() -> EvaluationReport {
        EvaluationReport(metrics: metrics, confusionMatrix: confusionMatrix)
    }
}

/// Standardized evaluation metrics result for regression models (G-019).
public struct RegressionEvaluation: Sendable, Codable, Equatable {
    /// Coefficient of determination R^2 in (-inf, 1].
    public let r2: Double
    /// Adjusted R^2 considering feature degrees of freedom, if specified.
    public let adjustedR2: Double?
    /// Mean Squared Error (MSE).
    public let mse: Double
    /// Root Mean Squared Error (RMSE).
    public let rmse: Double
    /// Mean Absolute Error (MAE).
    public let mae: Double
    /// Mean Absolute Percentage Error (MAPE) in percent [0, inf).
    public let mape: Double
    /// Explained variance score in (-inf, 1].
    public let explainedVariance: Double
    /// Flat dictionary of key evaluation metrics for tabular leaderboards and logging.
    public let metrics: [String: Double]

    /// Creates a new RegressionEvaluation instance.
    public init(
        r2: Double,
        adjustedR2: Double? = nil,
        mse: Double,
        rmse: Double,
        mae: Double,
        mape: Double,
        explainedVariance: Double,
        metrics: [String: Double]? = nil
    ) {
        self.r2 = r2
        self.adjustedR2 = adjustedR2
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.mape = mape
        self.explainedVariance = explainedVariance
        if let m = metrics {
            self.metrics = m
        } else {
            var dict: [String: Double] = [
                "r2": r2,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "explainedVariance": explainedVariance
            ]
            if let adj = adjustedR2 {
                dict["adjustedR2"] = adj
            }
            self.metrics = dict
        }
    }

    /// Converts this result into a standardized `EvaluationReport` from `SwiftML`.
    public func toEvaluationReport() -> EvaluationReport {
        EvaluationReport(metrics: metrics, confusionMatrix: nil)
    }
}

/// Unified Classifier & Regressor Evaluation Metrics Harness (G-019).
///
/// Computes comprehensive out-of-sample holdout metrics in a single standardized call,
/// eliminating manual multi-step boilerplate across consumer applications.
public enum EvaluationHarness {

    /// Computes full classification evaluation metrics (accuracy, macro/weighted precision, recall, F1, and confusion matrix).
    /// - Parameters:
    ///   - yTrue: Ground-truth target labels.
    ///   - yPred: Model predicted labels.
    /// - Returns: `ClassificationEvaluation` containing complete performance summary and confusion matrix.
    public static func evaluateClassification(yTrue: [Int], yPred: [Int]) -> ClassificationEvaluation {
        guard !yTrue.isEmpty, yTrue.count == yPred.count else {
            return ClassificationEvaluation(
                accuracy: 0.0,
                macroPrecision: 0.0,
                macroRecall: 0.0,
                macroF1: 0.0,
                weightedF1: 0.0,
                classLabels: [],
                confusionMatrix: []
            )
        }

        let report = Metrics.classificationReport(yTrue: yTrue, yPred: yPred)
        let labels = Array(Set(yTrue + yPred)).sorted()

        // Weighted F1
        let totalSupport = Double(yTrue.count)
        var weightedF1 = 0.0
        if totalSupport > 0 {
            for m in report.perClass {
                weightedF1 += m.f1 * (Double(m.support) / totalSupport)
            }
        }

        // Confusion Matrix
        let matrix = computeConfusionMatrix(yTrue: yTrue, yPred: yPred, labels: labels)

        var metricsDict: [String: Double] = [
            "accuracy": report.accuracy,
            "precision": report.macroPrecision,
            "recall": report.macroRecall,
            "f1": report.macroF1,
            "macroF1": report.macroF1,
            "weightedF1": weightedF1
        ]
        let mcc = Metrics.matthewsCorrelationCoefficient(yTrue: yTrue, yPred: yPred)
        if mcc.isFinite {
            metricsDict["mcc"] = mcc
        }

        return ClassificationEvaluation(
            accuracy: report.accuracy,
            macroPrecision: report.macroPrecision,
            macroRecall: report.macroRecall,
            macroF1: report.macroF1,
            weightedF1: weightedF1,
            classLabels: labels,
            confusionMatrix: matrix,
            metrics: metricsDict
        )
    }

    /// Overload for Double labels (e.g. classification target outputs encoded as Double).
    public static func evaluateClassification(yTrue: [Double], yPred: [Double]) -> ClassificationEvaluation {
        evaluateClassification(
            yTrue: yTrue.map { Int(round($0)) },
            yPred: yPred.map { Int(round($0)) }
        )
    }

    /// Overload for String class labels.
    public static func evaluateClassification(yTrue: [String], yPred: [String]) -> (
        evaluation: ClassificationEvaluation,
        labelMapping: [String: Int]
    ) {
        let uniqueLabels = Array(Set(yTrue + yPred)).sorted()
        let mapping = Dictionary(uniqueKeysWithValues: uniqueLabels.enumerated().map { ($0.element, $0.offset) })
        let intTrue = yTrue.compactMap { mapping[$0] }
        let intPred = yPred.compactMap { mapping[$0] }
        let eval = evaluateClassification(yTrue: intTrue, yPred: intPred)
        return (eval, mapping)
    }

    /// Computes full regression evaluation metrics (R^2, adjusted R^2, MSE, RMSE, MAE, MAPE, explained variance).
    /// - Parameters:
    ///   - yTrue: Ground-truth target continuous values.
    ///   - yPred: Model predicted continuous values.
    ///   - numFeatures: Optional number of predictor features for calculating adjusted R^2.
    /// - Returns: `RegressionEvaluation` containing complete regression performance metrics.
    public static func evaluateRegression(
        yTrue: [Double],
        yPred: [Double],
        numFeatures: Int? = nil
    ) -> RegressionEvaluation {
        guard !yTrue.isEmpty, yTrue.count == yPred.count else {
            return RegressionEvaluation(
                r2: 0.0,
                adjustedR2: nil,
                mse: 0.0,
                rmse: 0.0,
                mae: 0.0,
                mape: 0.0,
                explainedVariance: 0.0
            )
        }

        let r2 = Metrics.r2Score(yTrue: yTrue, yPred: yPred)
        let adjR2 = numFeatures.map { Metrics.adjustedR2Score(yTrue: yTrue, yPred: yPred, numFeatures: $0) }
        let mse = Metrics.meanSquaredError(yTrue: yTrue, yPred: yPred)
        let rmse = Metrics.rootMeanSquaredError(yTrue: yTrue, yPred: yPred)
        let mae = Metrics.meanAbsoluteError(yTrue: yTrue, yPred: yPred)
        let mape = Metrics.mape(yTrue: yTrue, yPred: yPred)
        let ev = Metrics.explainedVarianceScore(yTrue: yTrue, yPred: yPred)

        return RegressionEvaluation(
            r2: r2,
            adjustedR2: adjR2,
            mse: mse,
            rmse: rmse,
            mae: mae,
            mape: mape,
            explainedVariance: ev
        )
    }

    /// Computes a K x K confusion matrix where row index corresponds to actual label, and col index corresponds to predicted label.
    /// - Parameters:
    ///   - yTrue: Array of ground-truth class labels.
    ///   - yPred: Array of predicted class labels.
    ///   - labels: Optional explicit list of class labels defining the matrix ordering.
    /// - Returns: 2D array of counts with dimensions `[labels.count, labels.count]`.
    public static func computeConfusionMatrix(yTrue: [Int], yPred: [Int], labels: [Int]? = nil) -> [[Int]] {
        let classes = (labels ?? Array(Set(yTrue + yPred))).sorted()
        let k = classes.count
        guard k > 0 else { return [] }

        let labelToIdx = Dictionary(uniqueKeysWithValues: classes.enumerated().map { ($0.element, $0.offset) })
        var matrix = [[Int]](repeating: [Int](repeating: 0, count: k), count: k)

        for (actual, predicted) in zip(yTrue, yPred) {
            if let r = labelToIdx[actual], let c = labelToIdx[predicted] {
                matrix[r][c] += 1
            }
        }
        return matrix
    }
}
