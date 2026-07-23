import Foundation

/// Unified error hierarchy for the SwiftSci ecosystem.
public enum SwiftSciError: Error, CustomStringConvertible, Sendable {
    case dataError(String)
    case trainingError(String)
    case predictionError(String)
    case validationError(String)
    case ioError(String)
    case hardwareError(String)
    case pipelineError(String)

    public var description: String {
        switch self {
        case .dataError(let msg): return "[SwiftSci Data Error] \(msg)"
        case .trainingError(let msg): return "[SwiftSci Training Error] \(msg)"
        case .predictionError(let msg): return "[SwiftSci Prediction Error] \(msg)"
        case .validationError(let msg): return "[SwiftSci Validation Error] \(msg)"
        case .ioError(let msg): return "[SwiftSci IO Error] \(msg)"
        case .hardwareError(let msg): return "[SwiftSci Hardware Error] \(msg)"
        case .pipelineError(let msg): return "[SwiftSci Pipeline Error] \(msg)"
        }
    }
}
