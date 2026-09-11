import Foundation
#if canImport(NaturalLanguage)
import NaturalLanguage
#endif

/// Provides access to Apple's native pre-trained word and sentence embedding vector spaces (`NLEmbedding`).
public struct AppleNLEmbedding: Sendable {
    /// Supported language code enum for pre-trained word vector spaces.
    public enum Language: String, Sendable {
        case english = "en"
        case spanish = "es"
        case french = "fr"
        case german = "de"
        case italian = "it"
        case portuguese = "pt"
    }

    /// Configured language specification for embedding vectors.
    public let language: Language

    /// Creates an Apple NLEmbedding vector space instance.
    /// - Parameter language: Language vector space selection. Defaults to `.english`.
    public init(language: Language = .english) {
        self.language = language
    }

    /// Fetches the vector representation for a given word.
    /// - Parameters:
    ///   - word: Target word token string.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: Array of learned model parameters, or `nil` if the estimator is not yet fitted.
    public func vector(for word: String) throws -> [Double]? {
        #if canImport(NaturalLanguage)
        let nlLang = NLLanguage(rawValue: language.rawValue)
        guard let embedding = NLEmbedding.wordEmbedding(for: nlLang) else {
            return nil
        }
        return embedding.vector(for: word)
        #else
        throw NLPError.unavailableOnPlatform(feature: "AppleNLEmbedding")
        #endif
    }

    /// Calculates distance (1.0 - cosine similarity) between two words.
    /// - Parameters:
    ///   - word1: First word token string for comparison.
    ///   - word2: Second word token string for comparison.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: Computed scalar value, or `nil` if the model or parameter is uninitialized.
    public func distance(between word1: String, and word2: String) throws -> Double? {
        #if canImport(NaturalLanguage)
        let nlLang = NLLanguage(rawValue: language.rawValue)
        guard let embedding = NLEmbedding.wordEmbedding(for: nlLang) else {
            return nil
        }
        let distance = embedding.distance(between: word1, and: word2)
        return distance.isNaN ? nil : distance
        #else
        throw NLPError.unavailableOnPlatform(feature: "AppleNLEmbedding")
        #endif
    }

    /// Finds top-K nearest neighbor words for a given query word.
    /// - Parameters:
    ///   - word: Target word token string.
    ///   - maxCount: Maximum number of items or words to retain.
    /// - Throws: `SwiftMLError` or `NLPError` if vocabulary is uninitialized, files are unreadable, or models fail.
    /// - Returns: The computed [(word: String, distance: Double)]? result instance.
    public func nearestNeighbors(for word: String, maxCount: Int = 10) throws -> [(word: String, distance: Double)]? {
        #if canImport(NaturalLanguage)
        let nlLang = NLLanguage(rawValue: language.rawValue)
        guard let embedding = NLEmbedding.wordEmbedding(for: nlLang) else {
            return nil
        }
        let neighbors = embedding.neighbors(for: word, maximumCount: maxCount)
        return neighbors.map { (word: $0.0, distance: $0.1) }
        #else
        throw NLPError.unavailableOnPlatform(feature: "AppleNLEmbedding")
        #endif
    }
}
