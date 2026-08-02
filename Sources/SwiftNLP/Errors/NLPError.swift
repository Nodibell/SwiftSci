import Foundation

/// Errors that can occur during NLP processing.
public enum NLPError: Error, CustomStringConvertible, Sendable, Equatable {
    case unavailableOnPlatform(feature: String)
    case invalidInput(String)
    case modelNotFound(String)
    case emptyInput
    case invalidVocabulary
    case fittingRequired

    /// Human-readable error description message string.
    public var description: String {
        switch self {
        case .unavailableOnPlatform(let feature):
            return "NLP feature '\(feature)' is not available on the current platform or OS version."
        case .invalidInput(let message):
            return "Invalid input for NLP processing: \(message)"
        case .modelNotFound(let name):
            return "NLP model or resource '\(name)' could not be found."
        case .emptyInput:
            return "Input documents array is empty."
        case .invalidVocabulary:
            return "Failed to build a valid vocabulary from the provided documents."
        case .fittingRequired:
            return "The vectorizer must be fitted before calling transform."
        }
    }
}
