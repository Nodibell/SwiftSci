import Foundation

/// Utilities for LLM token context window management and prompt truncation.
public struct LLMContextWindow: Sendable {
    /// Maximum context window token capacity limit.
    public let maxTokens: Int
    /// Tokenizer instance for counting and truncating prompt tokens.
    public let tokenizer: any Tokenizer

    /// Creates an LLM context window manager instance.
    /// - Parameters:
    ///   - maxTokens: Maximum token capacity. Defaults to 4096.
    ///   - tokenizer: Tokenizer implementation. Defaults to `AppleWordTokenizer`.
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
