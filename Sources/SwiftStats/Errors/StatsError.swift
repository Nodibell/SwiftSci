/// Errors thrown by SwiftStats operations.
public enum StatsError: Error, Sendable, Equatable {

    // MARK: – Input validation
    case emptyInput
    case insufficientData(minimum: Int, got: Int)
    case sizeMismatch(sizeA: Int, sizeB: Int)
    case containsNaN
    case tooManySamples(limit: Int, got: Int)

    // MARK: – Parameter validation
    case invalidPercentile(Double)
    case invalidDDOF(Int)
    case invalidGroupCount(minimum: Int, got: Int)

    // MARK: – Numerical
    case divisionByZero(context: String)
    case negativeVariance
}

extension StatsError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .emptyInput:
            return "Input array must not be empty."
        case let .insufficientData(min, got):
            return "At least \(min) elements required, got \(got)."
        case let .sizeMismatch(a, b):
            return "Array sizes must match: got \(a) and \(b)."
        case .containsNaN:
            return "Input contains NaN values. Use .dropNaN() or .replaceNaN(with:) first."
        case let .tooManySamples(limit, got):
            return "Input size \(got) exceeds maximum supported sample limit of \(limit)."
        case let .invalidPercentile(q):
            return "Percentile q=\(q) is out of range [0, 1]."
        case let .invalidDDOF(ddof):
            return "Delta degrees of freedom \(ddof) must be ≥ 0."
        case let .invalidGroupCount(min, got):
            return "At least \(min) groups required, got \(got)."
        case let .divisionByZero(ctx):
            return "Division by zero in \(ctx)."
        case .negativeVariance:
            return "Computed variance is negative — possible floating-point underflow."
        }
    }
}
