import Foundation
import SwiftDataFrame
import SwiftML

extension DataFrame {
    /// Performs zero-shot text classification on a text column using ``ZeroShotTextClassifier``.
    ///
    /// - Parameters:
    ///   - textColumn: The name of the column containing the input text to classify.
    ///   - candidateLabels: The possible candidate class labels to predict.
    ///   - targetColumn: The name of the column to store predicted class labels (default `"predicted_label"`).
    ///   - multiLabel: Whether to classify each label independently using sigmoid (default `false`).
    ///   - classifier: The classifier instance to use (defaults to a new default ``ZeroShotTextClassifier``).
    /// - Throws: `SwiftMLError.columnNotFound` if `textColumn` does not exist, or `NLPError` if classification fails.
    /// - Returns: A new `DataFrame` containing the appended predicted label column.
    public func zeroShotClassify(
        column textColumn: String,
        candidateLabels: [String],
        targetColumn: String = "predicted_label",
        multiLabel: Bool = false,
        classifier: ZeroShotTextClassifier = ZeroShotTextClassifier()
    ) async throws -> DataFrame {
        guard let col = self[column: textColumn, as: String.self] else {
            throw SwiftMLError.columnNotFound(textColumn)
        }

        let texts = col.values.map { $0 ?? "" }
        let results = try await classifier.classifyBatch(
            texts: texts,
            candidateLabels: candidateLabels,
            multiLabel: multiLabel
        )
        let predictions = results.map { $0.predictedLabel }
        return try withColumn(targetColumn, column: TypedColumn<String>(name: targetColumn, values: predictions))
    }
}
