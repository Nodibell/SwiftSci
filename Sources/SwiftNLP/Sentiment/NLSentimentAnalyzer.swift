import Foundation
#if canImport(NaturalLanguage)
import NaturalLanguage
#endif

/// A sentiment analyzer wrapping Apple's machine learned OS model (`NLTagger(tagSchemes: [.sentimentScore])`).
public struct NLSentimentAnalyzer: Sendable {
    /// Creates a NaturalLanguage sentiment analyzer instance.
    public init() {}

    /// Evaluates text sentiment score (-1.0 to +1.0) using Apple's OS ML model.
    /// Returns 0.0 if text is neutral, whitespace/empty, or if no tag could be inferred by the OS tagger.
    /// For distinguishing missing results from neutral, use `scoreOptional(text:)`.
    /// - Parameter text: Input string.
    /// - Returns: Normalized sentiment score (-1.0 to +1.0).
    /// - Throws: `NLPError.unavailableOnPlatform` if `NaturalLanguage` is unavailable.
    public func score(text: String) throws -> Double {
        try scoreOptional(text: text) ?? 0.0
    }

    /// Evaluates text sentiment score (-1.0 to +1.0) using Apple's OS ML model, returning `nil`
    /// when the system model cannot infer a sentiment tag or the input is empty/whitespace.
    /// - Parameter text: Input string.
    /// - Returns: An optional score (-1.0 to +1.0), or `nil` if no sentiment score was produced.
    /// - Throws: `NLPError.unavailableOnPlatform` if `NaturalLanguage` is unavailable.
    public func scoreOptional(text: String) throws -> Double? {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return nil
        }

        #if canImport(NaturalLanguage)
        let tagger = NLTagger(tagSchemes: [.sentimentScore])
        tagger.string = text

        let (tag, _) = tagger.tag(at: text.startIndex, unit: .paragraph, scheme: .sentimentScore)
        if let scoreString = tag?.rawValue, let score = Double(scoreString) {
            return score
        }
        return nil
        #else
        throw NLPError.unavailableOnPlatform(feature: "NLSentimentAnalyzer")
        #endif
    }
}

