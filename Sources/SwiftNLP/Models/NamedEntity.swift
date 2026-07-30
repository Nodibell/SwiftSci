import Foundation

/// Represents a extracted Named Entity from text.
public struct NamedEntity: Sendable, Equatable, CustomStringConvertible {
    public enum EntityCategory: String, Sendable, Codable, Equatable {
        case personalName
        case placeName
        case organizationName
        case classifier
        case other
    }

    public let text: String
    public let category: EntityCategory
    public let range: Range<String.Index>
    public let confidence: Double

    public init(text: String, category: EntityCategory, range: Range<String.Index>, confidence: Double = 1.0) {
        self.text = text
        self.category = category
        self.range = range
        self.confidence = confidence
    }

    public var description: String {
        return "NamedEntity(\"\(text)\", category: .\(category.rawValue))"
    }
}
