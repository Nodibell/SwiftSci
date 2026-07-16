import Foundation

/// Errors thrown by SwiftForecast operations.
public enum ForecastError: Error, Sendable, Equatable {
    // MARK: - Input validation
    case emptyTimeSeries
    case insufficientLength(minimum: Int, got: Int)
    case containsNaN
    case containsInfinity

    // MARK: - Parameter validation
    case invalidAlpha(Double)
    case invalidBeta(Double)
    case invalidGamma(Double)
    case invalidPeriod(Int)
    case invalidAROrder(Int)
    case invalidDifferencing(Int)
    case invalidMAOrder(Int)
    case invalidHorizon(Int)

    // MARK: - Numerical
    case notFitted
    case convergenceFailed(iterations: Int)
    case singularMatrix
    case matrixDimensionMismatch(expectedRows: Int, expectedCols: Int, gotRows: Int, gotCols: Int)
}

extension ForecastError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .emptyTimeSeries:
            return "Time series must not be empty."
        case let .insufficientLength(minimum, got):
            return "Time series length is insufficient: minimum \(minimum) points required, got \(got)."
        case .containsNaN:
            return "Time series contains NaN values."
        case .containsInfinity:
            return "Time series contains infinite values."
        case let .invalidAlpha(alpha):
            return "Alpha parameter (\(alpha)) must be in (0, 1)."
        case let .invalidBeta(beta):
            return "Beta parameter (\(beta)) must be in (0, 1)."
        case let .invalidGamma(gamma):
            return "Gamma parameter (\(gamma)) must be in (0, 1)."
        case let .invalidPeriod(period):
            return "Seasonality period (\(period)) must be >= 2."
        case let .invalidAROrder(p):
            return "AR order p (\(p)) must be >= 0."
        case let .invalidDifferencing(d):
            return "Differencing parameter d (\(d)) must be >= 0."
        case let .invalidMAOrder(q):
            return "MA order q (\(q)) must be >= 0."
        case let .invalidHorizon(h):
            return "Forecast horizon (\(h)) must be >= 1."
        case .notFitted:
            return "The model has not been fitted yet."
        case let .convergenceFailed(iterations):
            return "Numerical solver failed to converge after \(iterations) iterations."
        case .singularMatrix:
            return "Matrix is singular (cannot invert)."
        case let .matrixDimensionMismatch(expectedRows, expectedCols, gotRows, gotCols):
            return "Matrix dimension mismatch: expected (\(expectedRows), \(expectedCols)), got (\(gotRows), \(gotCols))."
        }
    }
}
