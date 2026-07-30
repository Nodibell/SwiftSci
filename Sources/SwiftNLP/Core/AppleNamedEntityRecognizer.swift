import Foundation
#if canImport(NaturalLanguage)
import NaturalLanguage
#endif

/// Extracts Named Entities (Persons, Places, Organizations) from text using Apple's `NLTagger`.
public struct AppleNamedEntityRecognizer: Sendable {
    public init() {}

    /// Extracts named entities from input text.
    /// - Parameter text: Input string.
    /// - Returns: Array of `NamedEntity` objects.
    /// - Throws: `NLPError.unavailableOnPlatform` if `NaturalLanguage` is unavailable.
    public func extractEntities(from text: String) throws -> [NamedEntity] {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return []
        }

        #if canImport(NaturalLanguage)
        let tagger = NLTagger(tagSchemes: [.nameTypeOrLexicalClass])
        tagger.string = text
        var entities: [NamedEntity] = []

        let options: NLTagger.Options = [.omitPunctuation, .omitWhitespace, .joinNames]

        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .nameTypeOrLexicalClass, options: options) { tag, range in
            guard let tag = tag else { return true }

            let entityText = String(text[range])
            let category: NamedEntity.EntityCategory?

            switch tag {
            case .personalName:
                category = .personalName
            case .placeName:
                category = .placeName
            case .organizationName:
                category = .organizationName
            default:
                category = nil
            }

            if let cat = category {
                entities.append(NamedEntity(text: entityText, category: cat, range: range, confidence: 1.0))
            }
            return true
        }

        return entities
        #else
        throw NLPError.unavailableOnPlatform(feature: "AppleNamedEntityRecognizer")
        #endif
    }
}
