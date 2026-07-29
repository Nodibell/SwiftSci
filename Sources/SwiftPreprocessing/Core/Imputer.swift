import Foundation

/// Imputer fills missing values (represented by `Double.nan`) using a specified strategy.
public final class Imputer: PreprocessingTransformer, @unchecked Sendable {
    /// Represents strategy.
    public enum Strategy: Sendable, Equatable {
        case mean
        case median
        case mostFrequent
        case constant(Double)
    }
    
    /// The strategy.
    public let strategy: Strategy
    /// The statistics.
    public private(set) var statistics: [Double]?
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - strategy: The strategy.
    public init(strategy: Strategy = .mean) {
        self.strategy = strategy
    }
    
    /// Fits the imputer by calculating the chosen statistic for each column.
    public func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        
        let rowCount = data.count
        let colCount = data[0].count
        var computedStats = [Double](repeating: 0.0, count: colCount)
        
        for col in 0..<colCount {
            var colValues = [Double]()
            colValues.reserveCapacity(rowCount)
            
            for row in 0..<rowCount {
                guard data[row].count == colCount else {
                    throw PreprocessingError.dimensionMismatch(expected: colCount, got: data[row].count)
                }
                let val = data[row][col]
                if !val.isNaN {
                    colValues.append(val)
                }
            }
            
            if colValues.isEmpty {
                // Fallback for column with only missing values
                switch strategy {
                case .constant(let val):
                    computedStats[col] = val
                default:
                    computedStats[col] = 0.0
                }
                continue
            }
            
            switch strategy {
            case .mean:
                computedStats[col] = colValues.reduce(0.0, +) / Double(colValues.count)
            case .median:
                colValues.sort()
                let mid = colValues.count / 2
                if colValues.count % 2 == 0 {
                    computedStats[col] = (colValues[mid - 1] + colValues[mid]) / 2.0
                } else {
                    computedStats[col] = colValues[mid]
                }
            case .mostFrequent:
                var counts = [Double: Int]()
                for val in colValues {
                    counts[val, default: 0] += 1
                }
                var bestVal = colValues[0]
                var maxCount = 0
                for (val, count) in counts {
                    if count > maxCount {
                        maxCount = count
                        bestVal = val
                    } else if count == maxCount {
                        bestVal = min(bestVal, val)
                    }
                }
                computedStats[col] = bestVal
            case .constant(let val):
                computedStats[col] = val
            }
        }
        
        self.statistics = computedStats
    }
    
    /// Replaces `Double.nan` values with the fitted statistics.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard let stats = self.statistics else {
            throw PreprocessingError.fitNotCalled
        }
        guard !data.isEmpty else {
            return []
        }
        
        let colCount = stats.count
        var transformed = [[Double]]()
        transformed.reserveCapacity(data.count)
        
        for row in data {
            guard row.count == colCount else {
                throw PreprocessingError.dimensionMismatch(expected: colCount, got: row.count)
            }
            let imputedRow = (0..<colCount).map { col -> Double in
                let val = row[col]
                return val.isNaN ? stats[col] : val
            }
            transformed.append(imputedRow)
        }
        
        return transformed
    }
}
