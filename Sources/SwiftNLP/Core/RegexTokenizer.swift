import Foundation

/// A tokenizer that splits text based on a regular expression pattern.
public struct RegexTokenizer: Tokenizer, Sendable {
    /// The regular expression pattern.
    public let pattern: String
    /// Whether to match tokens directly (gaps = false) or split on gaps/separators (gaps = true).
    public let gaps: Bool

    /// Initializes a regex-pattern tokenizer instance.
    /// - Parameters:
    ///   - pattern: Regular expression pattern string. Defaults to `\w+`.
    ///   - gaps: If true, splits on matching gaps; if false, matches token substrings directly. Defaults to false.
    public init(pattern: String = #"\w+"#, gaps: Bool = false) {
        self.pattern = pattern
        self.gaps = gaps
    }

    /// Tokenizes input string according to the configured regex pattern.
    /// - Parameter text: Raw text string.
    /// - Returns: Array of token strings.
    public func tokenize(text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return [text]
        }

        let range = NSRange(text.startIndex..., in: text)

        if gaps {
            let matches = regex.matches(in: text, options: [], range: range)
            var tokens: [String] = []
            var lastIndex = text.startIndex

            for match in matches {
                if let matchRange = Range(match.range, in: text) {
                    let token = String(text[lastIndex..<matchRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                    if !token.isEmpty {
                        tokens.append(token)
                    }
                    lastIndex = matchRange.upperBound
                }
            }
            let remaining = String(text[lastIndex...]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !remaining.isEmpty {
                tokens.append(remaining)
            }
            return tokens
        } else {
            let matches = regex.matches(in: text, options: [], range: range)
            return matches.compactMap { match in
                guard let matchRange = Range(match.range, in: text) else { return nil }
                return String(text[matchRange])
            }
        }
    }

    /// Encodes input string into integer hashes.
    /// - Parameter text: Raw text string.
    /// - Returns: Array of integer token hashes.
    public func encode(text: String) -> [Int] {
        return tokenize(text: text).map { $0.hashValue }
    }

    /// Decodes token integer IDs back to string.
    /// - Parameter tokens: Array of integer token IDs.
    /// - Returns: Empty string for non-vocabulary hash encoders.
    public func decode(tokens: [Int]) -> String {
        return ""
    }
}
