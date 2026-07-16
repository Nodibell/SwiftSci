import Foundation

/// Custom errors thrown by the SwiftCluster module.
public enum ClusterError: Error, LocalizedError, Equatable, Sendable {
    case emptyInput
    case dimensionMismatch(expected: Int, got: Int)
    case fittingRequired
    case invalidParameter(String)
    case svdFailed(info: Int32)
    
    public var errorDescription: String? {
        switch self {
        case .emptyInput:
            return "Input dataset cannot be empty."
        case .dimensionMismatch(let expected, let got):
            return "Dimension mismatch: expected \(expected) features, but got \(got)."
        case .fittingRequired:
            return "The model must be fit before calling transform or predict."
        case .invalidParameter(let message):
            return "Invalid parameter: \(message)"
        case .svdFailed(let info):
            return "Singular Value Decomposition (SVD) failed with LAPACK info code: \(info)."
        }
    }
}
