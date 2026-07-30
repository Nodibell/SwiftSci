import Foundation
#if canImport(NaturalLanguage)
import NaturalLanguage
#endif

/// Detects the dominant language and language probabilities using Apple's `NLLanguageRecognizer`.
public struct AppleLanguageDetector: Sendable {
    public init() {}

    /// Detects dominant language code (e.g. "en", "uk", "fr") for the given text.
    /// - Parameter text: Input text.
    /// - Returns: Dominant language code string or nil if undetected.
    /// - Throws: `NLPError.unavailableOnPlatform` if `NaturalLanguage` is unavailable.
    public func detectLanguage(text: String) throws -> String? {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return nil
        }

        #if canImport(NaturalLanguage)
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        return recognizer.dominantLanguage?.rawValue
        #else
        throw NLPError.unavailableOnPlatform(feature: "AppleLanguageDetector")
        #endif
    }

    /// Detects language hypotheses with confidence scores.
    /// - Parameters:
    ///   - text: Input text.
    ///   - maxHypotheses: Maximum number of hypotheses to return.
    /// - Returns: Dictionary mapping language code strings to confidence scores.
    /// - Throws: `NLPError.unavailableOnPlatform` if `NaturalLanguage` is unavailable.
    public func detectLanguageHypotheses(text: String, maxHypotheses: Int = 5) throws -> [String: Double] {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return [:]
        }

        #if canImport(NaturalLanguage)
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        let hypotheses = recognizer.languageHypotheses(withMaximum: maxHypotheses)
        var result: [String: Double] = [:]
        for (lang, score) in hypotheses {
            result[lang.rawValue] = score
        }
        return result
        #else
        throw NLPError.unavailableOnPlatform(feature: "AppleLanguageDetector")
        #endif
    }
}
