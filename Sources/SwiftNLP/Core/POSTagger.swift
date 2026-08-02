import Foundation
#if canImport(NaturalLanguage)
import NaturalLanguage
#endif

/// A part-of-speech (POS) tagger.
public struct POSTagger: Sendable {
    /// Grammatical part-of-speech category tag.
    public enum POSTag: String, Sendable, Codable, Equatable {
        case noun
        case verb
        case adjective
        case adverb
        case pronoun
        case determiner
        case preposition
        case conjunction
        case interjection
        case numeral
        case punctuation
        case whitespace
        case other
    }

    /// A token paired with its predicted part-of-speech tag.
    public struct TaggedToken: Sendable, Equatable {
        /// The raw token word substring.
        public let token: String
        /// The assigned part-of-speech tag.
        public let tag: POSTag

        /// Initializes a tagged token.
        /// - Parameters:
        ///   - token: Token word string.
        ///   - tag: Assigned part-of-speech category.
        public init(token: String, tag: POSTag) {
            self.token = token
            self.tag = tag
        }
    }

    /// Creates a part-of-speech tagger instance.
    public init() {}

    /// Tags part-of-speech for each word in the input text.
    public func tag(text: String) -> [TaggedToken] {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return []
        }

        #if canImport(NaturalLanguage)
        let tagger = NLTagger(tagSchemes: [.lexicalClass])
        tagger.string = text
        var result: [TaggedToken] = []

        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .lexicalClass) { tag, range in
            let word = String(text[range])
            let posTag: POSTag
            if let tag = tag {
                switch tag {
                case .noun: posTag = .noun
                case .verb: posTag = .verb
                case .adjective: posTag = .adjective
                case .adverb: posTag = .adverb
                case .pronoun: posTag = .pronoun
                case .determiner: posTag = .determiner
                case .preposition: posTag = .preposition
                case .conjunction: posTag = .conjunction
                case .interjection: posTag = .interjection
                case .number: posTag = .numeral
                case .punctuation: posTag = .punctuation
                case .whitespace: posTag = .whitespace
                default: posTag = .other
                }
            } else {
                posTag = .other
            }
            result.append(TaggedToken(token: word, tag: posTag))
            return true
        }
        return result
        #else
        // Fallback rule-based POS tagger for non-Apple platforms
        let tokens = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        return tokens.map { token in
            let lower = token.lowercased()
            if Double(token) != nil {
                return TaggedToken(token: token, tag: .numeral)
            } else if ["the", "a", "an", "this", "that", "these", "those"].contains(lower) {
                return TaggedToken(token: token, tag: .determiner)
            } else if ["in", "on", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "off", "over", "under"].contains(lower) {
                return TaggedToken(token: token, tag: .preposition)
            } else if ["and", "but", "or", "nor", "so", "for", "yet"].contains(lower) {
                return TaggedToken(token: token, tag: .conjunction)
            } else if ["i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "their"].contains(lower) {
                return TaggedToken(token: token, tag: .pronoun)
            } else if lower.hasSuffix("ing") || lower.hasSuffix("ed") {
                return TaggedToken(token: token, tag: .verb)
            } else if lower.hasSuffix("ly") {
                return TaggedToken(token: token, tag: .adverb)
            } else if lower.hasSuffix("ful") || lower.hasSuffix("ous") || lower.hasSuffix("ive") || lower.hasSuffix("able") {
                return TaggedToken(token: token, tag: .adjective)
            } else {
                return TaggedToken(token: token, tag: .noun)
            }
        }
        #endif
    }
}
