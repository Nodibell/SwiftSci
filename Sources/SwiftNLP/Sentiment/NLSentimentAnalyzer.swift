import Foundation
#if canImport(NaturalLanguage)
import NaturalLanguage
#endif

/// A sentiment analyzer wrapping Apple's machine learned OS model (`NLTagger(tagSchemes: [.sentimentScore])`).
public struct NLSentimentAnalyzer: Sendable {
    /// Creates a NaturalLanguage sentiment analyzer instance.
    public init() {}

    /// Evaluates text sentiment score (-1.0 to +1.0) using Apple's OS ML model.
    /// - Parameter text: Input string.
    /// - Returns: Normalized sentiment score (-1.0 to +1.0).
    /// - Throws: `NLPError.unavailableOnPlatform` if `NaturalLanguage` is unavailable.
    public func score(text: String) throws -> Double {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return 0.0
        }

        #if canImport(NaturalLanguage)
        let tagger = NLTagger(tagSchemes: [.sentimentScore])
        tagger.string = text

        let (tag, _) = tagger.tag(at: text.startIndex, unit: .paragraph, scheme: .sentimentScore)
        if let scoreString = tag?.rawValue, let score = Double(scoreString) {
            return score
        }
        return 0.0
        #else
        throw NLPError.unavailableOnPlatform(feature: "NLSentimentAnalyzer")
        #endif
    }
}
