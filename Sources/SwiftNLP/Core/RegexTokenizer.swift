import Foundation

/// A tokenizer that splits text based on a regular expression pattern.
public struct RegexTokenizer: Tokenizer, Sendable {
    /// The regular expression pattern.
    public let pattern: String
    /// Whether to match tokens directly (gaps = false) or split on gaps/separators (gaps = true).
    public let gaps: Bool

    private let compiledRegex: NSRegularExpression?

    /// Initializes a regex-pattern tokenizer instance.
    /// - Parameters:
    ///   - pattern: Regular expression pattern string. Defaults to `\w+`.
    ///   - gaps: If true, splits on matching gaps; if false, matches token substrings directly. Defaults to false.
    public init(pattern: String = #"\w+"#, gaps: Bool = false) {
        self.pattern = pattern
        self.gaps = gaps
        self.compiledRegex = try? NSRegularExpression(pattern: pattern)
    }

    /// Initializes a regex-pattern tokenizer instance, validating that the pattern is a legal regular expression.
    /// - Parameters:
    ///   - pattern: Regular expression pattern string.
    ///   - gaps: If true, splits on matching gaps; if false, matches token substrings directly. Defaults to false.
    /// - Throws: `NLPError.invalidInput` if the pattern cannot be parsed as a valid regular expression.
    public init(validatingPattern pattern: String, gaps: Bool = false) throws {
        do {
            self.compiledRegex = try NSRegularExpression(pattern: pattern)
        } catch {
            throw NLPError.invalidInput("Invalid regular expression pattern '\(pattern)': \(error.localizedDescription)")
        }
        self.pattern = pattern
        self.gaps = gaps
    }

    /// Tokenizes input string according to the configured regex pattern.
    /// - Parameter text: Raw text string.
    /// - Returns: Array of token strings.
    public func tokenize(text: String) -> [String] {
        guard let regex = compiledRegex else {
            return []
        }
        return executeTokenize(with: regex, text: text)
    }

    /// Tokenizes input string according to the configured regex pattern, throwing if the regex is invalid.
    /// - Parameter text: Raw text string.
    /// - Returns: Array of token strings.
    /// - Throws: `NLPError.invalidInput` if the regex pattern is uncompiled or invalid.
    public func tokenizeThrowing(text: String) throws -> [String] {
        guard let regex = compiledRegex else {
            throw NLPError.invalidInput("Cannot tokenize: regex pattern '\(pattern)' is invalid.")
        }
        return executeTokenize(with: regex, text: text)
    }

    private func executeTokenize(with regex: NSRegularExpression, text: String) -> [String] {
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

    /// Encodes input string into deterministic 64-bit FNV-1a integer token hashes.
    /// - Parameter text: Raw text string.
    /// - Returns: Array of deterministic integer token hashes.
    public func encode(text: String) -> [Int] {
        return tokenize(text: text).map { fnv1aTokenHash($0) }
    }

    /// Decodes token integer IDs back to string.
    /// - Parameter tokens: Array of integer token IDs.
    /// - Returns: Empty string for non-vocabulary hash encoders.
    public func decode(tokens: [Int]) -> String {
        return ""
    }
}
