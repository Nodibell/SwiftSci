import Foundation

/// A tokenizer that splits text based on a regular expression pattern.
public struct RegexTokenizer: Tokenizer, Sendable {
    /// The regular expression pattern.
    public let pattern: String
    /// Whether to match tokens directly (gaps = false) or split on gaps/separators (gaps = true).
    public let gaps: Bool

    public init(pattern: String = #"\w+"#, gaps: Bool = false) {
        self.pattern = pattern
        self.gaps = gaps
    }

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

    public func encode(text: String) -> [Int] {
        return tokenize(text: text).map { $0.hashValue }
    }

    public func decode(tokens: [Int]) -> String {
        return ""
    }
}
