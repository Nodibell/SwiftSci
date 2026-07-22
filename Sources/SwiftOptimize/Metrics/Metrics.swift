import Foundation

// MARK: - Classification Report

/// Summary of per-class Precision, Recall, and F1 for a classifier.
public struct ClassificationReport: Sendable {
    public struct ClassMetrics: Sendable {
        public let label: Int
        public let precision: Double
        public let recall: Double
        public let f1: Double
        public let support: Int
    }

    public let perClass: [ClassMetrics]
    public let accuracy: Double
    public let macroPrecision: Double
    public let macroRecall: Double
    public let macroF1: Double
}

// MARK: - Metrics

/// A collection of evaluation metrics for classification and regression.
public enum Metrics {

    // MARK: Classification

    /// Accuracy score: fraction of correctly predicted labels.
    public static func accuracy(yTrue: [Int], yPred: [Int]) -> Double {
        guard !yTrue.isEmpty, yTrue.count == yPred.count else { return 0 }
        let correct = zip(yTrue, yPred).filter { $0 == $1 }.count
        return Double(correct) / Double(yTrue.count)
    }

    /// Precision for a given binary class label.
    public static func precision(yTrue: [Int], yPred: [Int], label: Int) -> Double {
        let tp = zip(yTrue, yPred).filter { $0.1 == label && $0.0 == label }.count
        let fp = zip(yTrue, yPred).filter { $0.1 == label && $0.0 != label }.count
        let denom = tp + fp
        return denom == 0 ? 0 : Double(tp) / Double(denom)
    }

    /// Recall for a given binary class label.
    public static func recall(yTrue: [Int], yPred: [Int], label: Int) -> Double {
        let tp = zip(yTrue, yPred).filter { $0.0 == label && $0.1 == label }.count
        let fn = zip(yTrue, yPred).filter { $0.0 == label && $0.1 != label }.count
        let denom = tp + fn
        return denom == 0 ? 0 : Double(tp) / Double(denom)
    }

    /// F1 score (harmonic mean of precision and recall) for a given class label.
    public static func f1Score(yTrue: [Int], yPred: [Int], label: Int) -> Double {
        let p = precision(yTrue: yTrue, yPred: yPred, label: label)
        let r = recall(yTrue: yTrue, yPred: yPred, label: label)
        let denom = p + r
        return denom == 0 ? 0 : 2 * p * r / denom
    }

    /// Full classification report: per-class Precision, Recall, F1 and macro averages.
    public static func classificationReport(yTrue: [Int], yPred: [Int]) -> ClassificationReport {
        let labels = Array(Set(yTrue + yPred)).sorted()
        var perClass = [ClassificationReport.ClassMetrics]()
        for label in labels {
            let p = precision(yTrue: yTrue, yPred: yPred, label: label)
            let r = recall(yTrue: yTrue, yPred: yPred, label: label)
            let f = f1Score(yTrue: yTrue, yPred: yPred, label: label)
            let support = yTrue.filter { $0 == label }.count
            perClass.append(.init(label: label, precision: p, recall: r, f1: f, support: support))
        }
        let acc = accuracy(yTrue: yTrue, yPred: yPred)
        let n = Double(labels.count)
        let macroPrecision = perClass.map(\.precision).reduce(0, +) / n
        let macroRecall    = perClass.map(\.recall).reduce(0, +) / n
        let macroF1        = perClass.map(\.f1).reduce(0, +) / n
        return ClassificationReport(
            perClass: perClass,
            accuracy: acc,
            macroPrecision: macroPrecision,
            macroRecall: macroRecall,
            macroF1: macroF1
        )
    }

    /// Balanced Accuracy: average recall per class.
    public static func balancedAccuracy(yTrue: [Int], yPred: [Int]) -> Double {
        let report = classificationReport(yTrue: yTrue, yPred: yPred)
        guard !report.perClass.isEmpty else { return 0 }
        let sumRecall = report.perClass.map(\.recall).reduce(0, +)
        return sumRecall / Double(report.perClass.count)
    }

    /// Matthews Correlation Coefficient (MCC) for binary classification.
    public static func matthewsCorrelationCoefficient(yTrue: [Int], yPred: [Int]) -> Double {
        guard yTrue.count == yPred.count, !yTrue.isEmpty else { return 0 }
        var tp = 0.0, tn = 0.0, fp = 0.0, fn = 0.0
        for (t, p) in zip(yTrue, yPred) {
            if t == 1 && p == 1 { tp += 1 }
            else if t == 0 && p == 0 { tn += 1 }
            else if t == 0 && p == 1 { fp += 1 }
            else if t == 1 && p == 0 { fn += 1 }
        }
        let num = (tp * tn) - (fp * fn)
        let den = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return den == 0 ? 0 : num / den
    }

    /// Cohen's Kappa score for inter-rater agreement.
    public static func cohenKappa(yTrue: [Int], yPred: [Int]) -> Double {
        guard yTrue.count == yPred.count, !yTrue.isEmpty else { return 0 }
        let po = accuracy(yTrue: yTrue, yPred: yPred)
        let labels = Array(Set(yTrue + yPred)).sorted()
        let n = Double(yTrue.count)
        
        var pe = 0.0
        for label in labels {
            let actualCount = Double(yTrue.filter { $0 == label }.count)
            let predCount = Double(yPred.filter { $0 == label }.count)
            pe += (actualCount / n) * (predCount / n)
        }
        
        let den = 1.0 - pe
        return den == 0 ? 1.0 : (po - pe) / den
    }

    // MARK: Probability & Curve Metrics

    /// Logarithmic Loss (Binary Cross-Entropy).
    public static func logLoss(yTrue: [Int], yScore: [Double], eps: Double = 1e-15) -> Double {
        guard yTrue.count == yScore.count, !yTrue.isEmpty else { return 0 }
        var loss = 0.0
        for (y, s) in zip(yTrue, yScore) {
            let p = max(eps, min(1.0 - eps, s))
            let yD = Double(y)
            loss += -(yD * log(p) + (1.0 - yD) * log(1.0 - p))
        }
        return loss / Double(yTrue.count)
    }

    /// Brier Score: Mean Squared Error between true binary labels and predicted probabilities.
    public static func brierScore(yTrue: [Int], yScore: [Double]) -> Double {
        guard yTrue.count == yScore.count, !yTrue.isEmpty else { return 0 }
        let sumSq = zip(yTrue, yScore).map { pow(Double($0) - $1, 2) }.reduce(0, +)
        return sumSq / Double(yTrue.count)
    }

    /// Computes False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for ROC curve.
    public static func rocCurve(yTrue: [Int], yScore: [Double]) -> [(fpr: Double, tpr: Double, threshold: Double)] {
        guard yTrue.count == yScore.count, !yTrue.isEmpty else { return [] }
        let paired = zip(yTrue, yScore).sorted { $0.1 > $1.1 }
        
        let totalP = Double(yTrue.filter { $0 == 1 }.count)
        let totalN = Double(yTrue.filter { $0 == 0 }.count)
        guard totalP > 0, totalN > 0 else { return [] }

        var points: [(fpr: Double, tpr: Double, threshold: Double)] = [(0.0, 0.0, Double.infinity)]
        var tp = 0.0, fp = 0.0

        for i in 0..<paired.count {
            if paired[i].0 == 1 { tp += 1 } else { fp += 1 }
            
            if i == paired.count - 1 || paired[i].1 != paired[i + 1].1 {
                points.append((fpr: fp / totalN, tpr: tp / totalP, threshold: paired[i].1))
            }
        }
        return points
    }

    /// Computes Area Under the ROC Curve (ROC-AUC) via trapezoidal integration.
    public static func rocAUC(yTrue: [Int], yScore: [Double]) -> Double {
        let points = rocCurve(yTrue: yTrue, yScore: yScore)
        guard points.count >= 2 else { return 0 }
        var auc = 0.0
        for i in 1..<points.count {
            let dx = points[i].fpr - points[i - 1].fpr
            let avgY = (points[i].tpr + points[i - 1].tpr) / 2.0
            auc += dx * avgY
        }
        return auc
    }

    /// Computes Precision, Recall, and thresholds for Precision-Recall curve.
    public static func prCurve(yTrue: [Int], yScore: [Double]) -> [(precision: Double, recall: Double, threshold: Double)] {
        guard yTrue.count == yScore.count, !yTrue.isEmpty else { return [] }
        let paired = zip(yTrue, yScore).sorted { $0.1 > $1.1 }
        let totalP = Double(yTrue.filter { $0 == 1 }.count)
        guard totalP > 0 else { return [] }

        var points: [(precision: Double, recall: Double, threshold: Double)] = []
        var tp = 0.0, fp = 0.0

        for i in 0..<paired.count {
            if paired[i].0 == 1 { tp += 1 } else { fp += 1 }
            
            if i == paired.count - 1 || paired[i].1 != paired[i + 1].1 {
                let p = tp / (tp + fp)
                let r = tp / totalP
                points.append((precision: p, recall: r, threshold: paired[i].1))
            }
        }
        return points
    }

    // MARK: Regression

    /// Mean Squared Error.
    public static func meanSquaredError(yTrue: [Double], yPred: [Double]) -> Double {
        guard !yTrue.isEmpty, yTrue.count == yPred.count else { return 0 }
        return zip(yTrue, yPred).map { pow($0 - $1, 2) }.reduce(0, +) / Double(yTrue.count)
    }

    /// Root Mean Squared Error.
    public static func rootMeanSquaredError(yTrue: [Double], yPred: [Double]) -> Double {
        return sqrt(meanSquaredError(yTrue: yTrue, yPred: yPred))
    }

    /// Mean Absolute Error.
    public static func meanAbsoluteError(yTrue: [Double], yPred: [Double]) -> Double {
        guard !yTrue.isEmpty, yTrue.count == yPred.count else { return 0 }
        return zip(yTrue, yPred).map { abs($0 - $1) }.reduce(0, +) / Double(yTrue.count)
    }

    /// R² (coefficient of determination): fraction of variance explained.
    public static func r2Score(yTrue: [Double], yPred: [Double]) -> Double {
        guard !yTrue.isEmpty, yTrue.count == yPred.count else { return 0 }
        let mean = yTrue.reduce(0, +) / Double(yTrue.count)
        let ssTot = yTrue.map { pow($0 - mean, 2) }.reduce(0, +)
        let ssRes = zip(yTrue, yPred).map { pow($0 - $1, 2) }.reduce(0, +)
        return ssTot == 0 ? 0 : 1.0 - ssRes / ssTot
    }
}

// MARK: - Pretty Printing Helper

extension ClassificationReport {
    public func prettyPrint() {
        let labelHeader = "Label".padding(toLength: 10, withPad: " ", startingAt: 0)
        let precisionHeader = "Precision".padding(toLength: 10, withPad: " ", startingAt: 0)
        let recallHeader = "Recall".padding(toLength: 10, withPad: " ", startingAt: 0)
        let f1Header = "F1".padding(toLength: 10, withPad: " ", startingAt: 0)
        let supportHeader = "Support".padding(toLength: 10, withPad: " ", startingAt: 0)
        print("\(labelHeader) \(precisionHeader) \(recallHeader) \(f1Header) \(supportHeader)")
        print(String(repeating: "-", count: 54))
        
        for m in perClass {
            let labelStr = String(format: "%-10d", m.label)
            let precisionStr = String(format: "%10.4f", m.precision)
            let recallStr = String(format: "%10.4f", m.recall)
            let f1Str = String(format: "%10.4f", m.f1)
            let supportStr = String(format: "%10d", m.support)
            print("\(labelStr) \(precisionStr) \(recallStr) \(f1Str) \(supportStr)")
        }
        
        print(String(repeating: "-", count: 54))
        
        let macroLabel = "Macro Avg".padding(toLength: 10, withPad: " ", startingAt: 0)
        let macroPrecisionStr = String(format: "%10.4f", macroPrecision)
        let macroRecallStr = String(format: "%10.4f", macroRecall)
        let macroF1Str = String(format: "%10.4f", macroF1)
        print("\(macroLabel) \(macroPrecisionStr) \(macroRecallStr) \(macroF1Str)")
        
        print(String(format: "Accuracy: %.4f", accuracy))
    }
}
