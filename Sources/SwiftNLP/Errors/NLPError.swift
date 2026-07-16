import Foundation

/// Custom errors thrown by the SwiftNLP module.
public enum NLPError: Error, LocalizedError, Equatable {
    case emptyInput
    case fittingRequired
    case invalidVocabulary
    
    public var errorDescription: String? {
        switch self {
        case .emptyInput:
            return "Input corpus or document list cannot be empty."
        case .fittingRequired:
            return "The vectorizer must be fit on a corpus before calling transform."
        case .invalidVocabulary:
            return "The vocabulary is empty or invalid."
        }
    }
}
