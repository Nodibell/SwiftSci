import Foundation

/// MissingValueIndicator generates a binary mask indicating the presence of missing values (NaNs).
public final class MissingValueIndicator: PreprocessingTransformer, @unchecked Sendable {
    /// Represents features option.
    public enum FeaturesOption: String, Sendable, Codable {
        case all
        case missingOnly = "missing-only"
    }
    
    /// The features.
    public let features: FeaturesOption
    /// The indicator indices.
    public private(set) var indicatorIndices: [Int]?
    /// The fit feature count.
    public private(set) var fitFeatureCount: Int?
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - features: The features.
    public init(features: FeaturesOption = .missingOnly) {
        self.features = features
    }
    
    /// Fits the indicator by identifying which columns contain missing values (if features is .missingOnly).
    public func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        
        let rowCount = data.count
        let colCount = data[0].count
        self.fitFeatureCount = colCount
        var activeCols = [Int]()
        
        for col in 0..<colCount {
            var hasMissing = false
            for row in 0..<rowCount {
                guard data[row].count == colCount else {
                    throw PreprocessingError.dimensionMismatch(expected: colCount, got: data[row].count)
                }
                if data[row][col].isNaN {
                    hasMissing = true
                    break
                }
            }
            
            if features == .all || hasMissing {
                activeCols.append(col)
            }
        }
        
        self.indicatorIndices = activeCols
    }
    
    /// Transforms the dataset into a binary indicator matrix.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard let indices = self.indicatorIndices, let expectedCols = self.fitFeatureCount else {
            throw PreprocessingError.fitNotCalled
        }
        guard !data.isEmpty else {
            return []
        }
        
        var transformed = [[Double]]()
        transformed.reserveCapacity(data.count)
        
        for row in data {
            guard row.count == expectedCols else {
                throw PreprocessingError.dimensionMismatch(expected: expectedCols, got: row.count)
            }
            let indicatorRow = indices.map { col -> Double in
                row[col].isNaN ? 1.0 : 0.0
            }
            transformed.append(indicatorRow)
        }
        
        return transformed
    }
}
