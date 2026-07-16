import Foundation

/// Errors thrown by the preprocessing and encoding components.
public enum PreprocessingError: LocalizedError, Sendable {
    case emptyInput
    case fitNotCalled
    case dimensionMismatch(expected: Int, got: Int)
    case unknownCategory(String)
    case constantColumn(index: Int)
    
    public var errorDescription: String? {
        switch self {
        case .emptyInput:
            return "Input data cannot be empty."
        case .fitNotCalled:
            return "Estimator must be fitted before transforming data."
        case .dimensionMismatch(let expected, let got):
            return "Dimension mismatch: expected \(expected) features, but got \(got)."
        case .unknownCategory(let category):
            return "Unknown category encountered during transformation: '\(category)'."
        case .constantColumn(let index):
            return "Column at index \(index) has zero variance (constant values)."
        }
    }
}
