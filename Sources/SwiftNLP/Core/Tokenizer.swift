import Foundation
@_exported import SwiftDataFrame

/// A unified protocol for tokenizing and encoding text.
public protocol Tokenizer: Sendable {
    /// Splits the given text into an array of string tokens.
    /// - Parameter text: The input string.
    /// - Returns: An array of token strings.
    func tokenize(text: String) -> [String]
    
    /// Encodes the given text into a sequence of token IDs.
    /// - Parameter text: The input string.
    /// - Returns: An array of token ID integers.
    func encode(text: String) -> [Int]
    
    /// Decodes a sequence of token IDs back into a reconstructed string.
    /// - Parameter tokens: An array of token ID integers.
    /// - Returns: The reconstructed string.
    func decode(tokens: [Int]) -> String

    /// Splits text into sentence tokens.
    /// - Parameter text: Input text string.
    /// - Returns: Array of sentence strings.
    func tokenizeSentences(text: String) -> [String]
}

extension Tokenizer {
    /// Default protocol extension implementation delegating to `SentenceTokenizer`.
    /// - Parameter text: Input text document.
    /// - Returns: Array of sentence strings.
    public func tokenizeSentences(text: String) -> [String] {
        return SentenceTokenizer().tokenize(text: text)
    }
}

