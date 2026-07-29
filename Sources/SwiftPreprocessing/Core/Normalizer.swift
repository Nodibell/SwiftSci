import Foundation

/// Normalizer scales individual samples (rows) to have unit norm.
public final class Normalizer: PreprocessingTransformer, @unchecked Sendable {
    /// Represents norm type.
    public enum NormType: Sendable {
        case l1
        case l2
        case max
    }
    
    /// The norm.
    public let norm: NormType
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - norm: The norm.
    public init(norm: NormType = .l2) {
        self.norm = norm
    }
    
    /// Normalizer is stateless, so fit is a no-op that just checks input validity.
    public func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
    }
    
    /// Normalizes each row of the dataset.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard !data.isEmpty else {
            return []
        }
        
        let colCount = data[0].count
        var transformed = [[Double]]()
        transformed.reserveCapacity(data.count)
        
        for row in data {
            guard row.count == colCount else {
                throw PreprocessingError.dimensionMismatch(expected: colCount, got: row.count)
            }
            
            var normVal = 0.0
            switch norm {
            case .l1:
                normVal = row.reduce(0.0) { $0 + abs($1) }
            case .l2:
                let sumSq = row.reduce(0.0) { $0 + ($1 * $1) }
                normVal = sumSq.squareRoot()
            case .max:
                normVal = row.reduce(0.0) { max($0, abs($1)) }
            }
            
            if normVal < 1e-12 {
                transformed.append([Double](repeating: 0.0, count: colCount))
            } else {
                let normRow = row.map { $0 / normVal }
                transformed.append(normRow)
            }
        }
        
        return transformed
    }
}
