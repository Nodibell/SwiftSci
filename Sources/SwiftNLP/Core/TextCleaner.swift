import Foundation

/// High-performance text cleaner and normalizer for NLP pipelines.
public struct TextCleaner: Sendable {
    
    public init() {}
    
    /// Normalizes and cleans raw text according to configurable pipeline rules.
    /// - Parameters:
    ///   - text: Raw input text.
    ///   - stripURLs: Whether to remove HTTP/HTTPS/WWW URLs. Defaults to true.
    ///   - stripEmails: Whether to remove email addresses. Defaults to true.
    ///   - stripHTML: Whether to remove HTML/XML tags. Defaults to true.
    ///   - stripPunctuation: Whether to replace punctuation with spaces. Defaults to false.
    ///   - lowercase: Whether to convert text to lowercase. Defaults to true.
    /// - Returns: Cleaned and normalized text string.
    public static func clean(
        _ text: String,
        stripURLs: Bool = true,
        stripEmails: Bool = true,
        stripHTML: Bool = true,
        stripPunctuation: Bool = false,
        lowercase: Bool = true
    ) -> String {
        var current = text
        
        if stripURLs && (current.contains("http") || current.contains("www.")) {
            current = self.stripURLs(current)
        }
        
        if stripEmails && current.contains("@") {
            current = self.stripEmails(current)
        }
        
        if stripHTML && (current.contains("<") && current.contains(">")) {
            current = self.stripHTML(current)
        }
        
        if lowercase {
            current = current.lowercased()
        }
        
        if stripPunctuation {
            current = self.stripPunctuation(current)
        }
        
        return self.normalizeWhitespace(current)
    }
    
    /// Strips URL patterns from text.
    public static func stripURLs(_ text: String) -> String {
        guard text.contains("http") || text.contains("www.") else { return text }
        let pattern = "(?i)https?://\\S+|www\\.\\S+"
        return text.replacingOccurrences(of: pattern, with: " ", options: .regularExpression)
    }
    
    /// Strips email addresses from text.
    public static func stripEmails(_ text: String) -> String {
        guard text.contains("@") else { return text }
        let pattern = "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
        return text.replacingOccurrences(of: pattern, with: " ", options: .regularExpression)
    }
    
    /// Strips HTML and XML tags from text.
    public static func stripHTML(_ text: String) -> String {
        guard text.contains("<") && text.contains(">") else { return text }
        let pattern = "<[^>]+>"
        return text.replacingOccurrences(of: pattern, with: " ", options: .regularExpression)
    }
    
    /// Replaces non-letter and non-digit characters with whitespace.
    public static func stripPunctuation(_ text: String) -> String {
        var scalars: [Unicode.Scalar] = []
        scalars.reserveCapacity(text.unicodeScalars.count)
        for s in text.unicodeScalars {
            if Character(s).isLetter || Character(s).isNumber || Character(s).isWhitespace {
                scalars.append(s)
            } else {
                scalars.append(" ")
            }
        }
        return String(String.UnicodeScalarView(scalars))
    }
    
    /// Collapses consecutive whitespace into single spaces and trims leading/trailing spaces.
    public static func normalizeWhitespace(_ text: String) -> String {
        let tokens = text.split(separator: " ", omittingEmptySubsequences: true)
        return tokens.joined(separator: " ")
    }
}
