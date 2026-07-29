import Foundation

/// Utilities for standard text normalization.
public struct TextNormalizer: Sendable {
    /// The lowercase.
    public var lowercase: Bool
    /// The remove punctuation.
    public var removePunctuation: Bool

    /// Creates a new instance.
    /// - Parameters:
    ///   - lowercase: The lowercase.
    ///   - removePunctuation: The remove punctuation.
    public init(lowercase: Bool = true, removePunctuation: Bool = true) {
        self.lowercase = lowercase
        self.removePunctuation = removePunctuation
    }

    /// Normalizes input text according to configured options and Unicode normalization.
    public func normalize(_ text: String) -> String {
        var result = text.precomposedStringWithCanonicalMapping
        if lowercase {
            result = result.lowercased()
        }
        if removePunctuation {
            result = result.components(separatedBy: .punctuationCharacters).joined()
        }
        return result
    }
}
