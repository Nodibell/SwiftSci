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
