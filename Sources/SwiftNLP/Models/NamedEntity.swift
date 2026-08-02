import Foundation

/// Represents an extracted Named Entity from text.
public struct NamedEntity: Sendable, Equatable, CustomStringConvertible {
    /// Categorical classification category for named entities.
    public enum EntityCategory: String, Sendable, Codable, Equatable {
        case personalName
        case placeName
        case organizationName
        case classifier
        case other
    }

    /// Extracted named entity text substring.
    public let text: String
    /// Assigned entity classification category.
    public let category: EntityCategory
    /// Character index range within the source document string.
    public let range: Range<String.Index>
    /// Model confidence probability score (0.0 to 1.0).
    public let confidence: Double

    /// Creates an extracted named entity value object.
    /// - Parameters:
    ///   - text: Entity text string.
    ///   - category: Entity classification category.
    ///   - range: String range within the source text.
    ///   - confidence: Model confidence score. Defaults to 1.0.
    public init(text: String, category: EntityCategory, range: Range<String.Index>, confidence: Double = 1.0) {
        self.text = text
        self.category = category
        self.range = range
        self.confidence = confidence
    }

    /// Human-readable textual representation.
    public var description: String {
        return "NamedEntity(\"\(text)\", category: .\(category.rawValue))"
    }
}
