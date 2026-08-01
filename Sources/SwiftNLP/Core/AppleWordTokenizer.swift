import Foundation
#if canImport(NaturalLanguage)
import NaturalLanguage
#endif

/// A word tokenizer wrapping Apple's `NaturalLanguage.NLTokenizer(unit: .word)` with multi-lingual support.
public struct AppleWordTokenizer: Tokenizer, Sendable {
    public init() {}

    /// Tokenizes input text into word tokens using Apple's linguistic boundary detection.
    public func tokenize(text: String) -> [String] {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return []
        }

        #if canImport(NaturalLanguage)
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        var tokens: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let word = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !word.isEmpty {
                tokens.append(word)
            }
            return true
        }
        return tokens
        #else
        // Fallback simple word tokenizer for non-Apple platforms
        return text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        #endif
    }

    /// Deterministic 64-bit FNV-1a hashing for persistent token encoding across process restarts.
    private func fnv1aHash(_ string: String) -> Int {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x100000001b3
        }
        return Int(truncatingIfNeeded: hash)
    }

    public func encode(text: String) -> [Int] {
        return tokenize(text: text).map { fnv1aHash($0) }
    }

    public func decode(tokens: [Int]) -> String {
        return ""
    }
}
