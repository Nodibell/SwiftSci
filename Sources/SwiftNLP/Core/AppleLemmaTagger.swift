import Foundation
#if canImport(NaturalLanguage)
import NaturalLanguage
#endif

/// A lemmatizer using Apple's `NLTagger(tagSchemes: [.lemma])` for dictionary canonical form extraction.
public struct AppleLemmaTagger: Sendable {
    public init() {}

    /// Lemmatizes input text into canonical base word forms.
    /// - Parameter text: Input string.
    /// - Returns: Array of lemmatized strings.
    /// - Throws: `NLPError.unavailableOnPlatform` if `NaturalLanguage` is unavailable.
    public func lemmatize(text: String) throws -> [String] {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return []
        }

        #if canImport(NaturalLanguage)
        let tagger = NLTagger(tagSchemes: [.lemma])
        tagger.string = text
        var lemmas: [String] = []

        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .lemma) { tag, range in
            let word = tag?.rawValue ?? String(text[range])
            let cleaned = word.trimmingCharacters(in: .whitespacesAndNewlines)
            if !cleaned.isEmpty {
                lemmas.append(cleaned)
            }
            return true
        }
        return lemmas
        #else
        throw NLPError.unavailableOnPlatform(feature: "AppleLemmaTagger")
        #endif
    }
}
