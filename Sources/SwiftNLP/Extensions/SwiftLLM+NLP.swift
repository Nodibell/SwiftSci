import Foundation

/// Utilities for LLM token context window management and prompt truncation.
public struct LLMContextWindow: Sendable {
    public let maxTokens: Int
    public let tokenizer: any Tokenizer

    public init(maxTokens: Int = 4096, tokenizer: any Tokenizer = AppleWordTokenizer()) {
        self.maxTokens = maxTokens
        self.tokenizer = tokenizer
    }

    /// Estimates total token count for input prompt.
    public func countTokens(in text: String) -> Int {
        return tokenizer.tokenize(text: text).count
    }

    /// Truncates text to fit within the maximum allowed token count.
    public func truncate(text: String, maxTokens limit: Int? = nil) -> String {
        let maxAllowed = limit ?? self.maxTokens
        let tokens = tokenizer.tokenize(text: text)
        if tokens.count <= maxAllowed {
            return text
        }
        let truncatedTokens = Array(tokens.prefix(maxAllowed))
        return truncatedTokens.joined(separator: " ")
    }
}
