import Foundation
import SwiftDataFrame

/// A unified pipeline combining text tokenization, vectorization, and Naive Bayes text classification.
@available(*, deprecated, message: "TextPipeline uses deprecated MultinomialNaiveBayes. Use NaiveBayesClassifier directly instead.")
public actor TextPipeline {
    /// Internal TF-IDF document vectorizer instance.
    public let vectorizer: TFIDFVectorizer
    /// Internal Multinomial Naive Bayes text classifier instance.
    public private(set) var classifier: MultinomialNaiveBayes

    /// Initializes a text classification pipeline.
    /// - Parameter alpha: Laplace smoothing parameter for Naive Bayes. Defaults to 1.0.
    public init(alpha: Double = 1.0) {
        self.vectorizer = TFIDFVectorizer()
        self.classifier = MultinomialNaiveBayes(alpha: alpha)
    }

    /// Fits the text pipeline on raw document strings and target labels.
    /// - Parameters:
    ///   - documents: Collection of text documents to process.
    ///   - labels: Array of discrete class labels.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    public func fit(documents: [String], labels: [String]) async throws {
        try await vectorizer.fit(documents)
        let X = try await vectorizer.transform(documents)
        classifier.fit(X: X, y: labels)
    }

    /// Predicts the class label for a single document string.
    /// - Parameters:
    ///   - document: Single text document string.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: Textual string representation, or `nil` if absent.
    public func predict(document: String) async throws -> String? {
        let x = try await vectorizer.transform([document])
        guard let first = x.first else { return nil }
        return classifier.predict(x: first)
    }

    /// Predicts class labels for an array of document strings.
    /// - Parameters:
    ///   - documents: Collection of text documents to process.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: Array of feature names, column identifiers, or tokens.
    public func predict(documents: [String]) async throws -> [String] {
        let X = try await vectorizer.transform(documents)
        return classifier.predict(X: X)
    }
}


