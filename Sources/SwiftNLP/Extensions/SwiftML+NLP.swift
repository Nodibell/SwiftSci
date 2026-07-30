import Foundation
import SwiftDataFrame

/// A unified pipeline combining text tokenization, vectorization, and Naive Bayes text classification.
public actor TextPipeline {
    public let vectorizer: TFIDFVectorizer
    public private(set) var classifier: MultinomialNaiveBayes

    public init(alpha: Double = 1.0) {
        self.vectorizer = TFIDFVectorizer()
        self.classifier = MultinomialNaiveBayes(alpha: alpha)
    }

    /// Fits the text pipeline on raw document strings and target labels.
    public func fit(documents: [String], labels: [String]) async throws {
        try await vectorizer.fit(documents)
        let X = try await vectorizer.transform(documents)
        classifier.fit(X: X, y: labels)
    }

    /// Predicts the class label for a single document string.
    public func predict(document: String) async throws -> String? {
        let x = try await vectorizer.transform([document])
        guard let first = x.first else { return nil }
        return classifier.predict(x: first)
    }

    /// Predicts class labels for an array of document strings.
    public func predict(documents: [String]) async throws -> [String] {
        let X = try await vectorizer.transform(documents)
        return classifier.predict(X: X)
    }
}
