import Foundation
#if canImport(NaturalLanguage)
import NaturalLanguage
#endif

/// A sentence tokenizer splitting input text into individual sentences.
public struct SentenceTokenizer: Tokenizer, Sendable {
    /// Creates a sentence tokenizer instance.
    public init() {}

    /// Splits text into sentence strings.
    public func tokenize(text: String) -> [String] {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return []
        }

        #if canImport(NaturalLanguage)
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        var sentences: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty {
                sentences.append(sentence)
            }
            return true
        }
        return sentences
        #else
        // Fallback rule-based sentence tokenization for non-Apple platforms
        let pattern = #"(?<=[.!?])\s+"#
        if let regex = try? NSRegularExpression(pattern: pattern) {
            let range = NSRange(text.startIndex..., in: text)
            let matches = regex.matches(in: text, options: [], range: range)
            var sentences: [String] = []
            var lastIndex = text.startIndex

            for match in matches {
                if let matchRange = Range(match.range, in: text) {
                    let sentence = String(text[lastIndex..<matchRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                    if !sentence.isEmpty {
                        sentences.append(sentence)
                    }
                    lastIndex = matchRange.upperBound
                }
            }
            let remaining = String(text[lastIndex...]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !remaining.isEmpty {
                sentences.append(remaining)
            }
            return sentences
        }
        return [text]
        #endif
    }

    /// Encodes input string into integer hashes.
    /// - Parameter text: Raw text string.
    /// - Returns: Array of integer sentence hashes.
    public func encode(text: String) -> [Int] {
        return tokenize(text: text).map { $0.hashValue }
    }

    /// Decodes token integer IDs back to string.
    /// - Parameter tokens: Array of integer token IDs.
    /// - Returns: Empty string for non-vocabulary hash encoders.
    public func decode(tokens: [Int]) -> String {
        return ""
    }

    /// Splits text into individual sentences.
    /// - Parameter text: Input text document.
    /// - Returns: Array of sentence strings.
    public func tokenizeSentences(text: String) -> [String] {
        return tokenize(text: text)
    }
}
