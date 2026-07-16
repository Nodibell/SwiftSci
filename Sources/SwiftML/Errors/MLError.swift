import Foundation

/// Errors thrown by the machine learning models.
public enum MLError: LocalizedError, Sendable {
    case emptyInput
    case dimensionMismatch(expected: Int, got: Int)
    case modelNotFitted
    case trainingFailed(String)
    case invalidParameter(String)
    case notFitted
    
    public var errorDescription: String? {
        switch self {
        case .emptyInput:
            return "Input X or y cannot be empty."
        case .dimensionMismatch(let expected, let got):
            return "Dimension mismatch: expected \(expected) features, but got \(got)."
        case .modelNotFitted, .notFitted:
            return "Model must be fitted before making predictions."
        case .trainingFailed(let reason):
            return "Model training failed: \(reason)"
        case .invalidParameter(let reason):
            return "Invalid parameter: \(reason)"
        }
    }
}
