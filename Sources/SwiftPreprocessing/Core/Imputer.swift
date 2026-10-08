import Foundation

/// Imputer fills missing values (represented by `Double.nan`) using a specified strategy.
public struct Imputer: PreprocessingTransformer, Sendable {
    /// Fits and transforms compact columns directly in prepared pipelines.
    public var supportsNativePreparedBatches: Bool { true }

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
    /// - Parameters:
    ///   - data: Raw input data array or matrix for transformation.
    /// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
    public mutating func fit(_ data: [[Double]]) throws {

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
            
            computedStats[col] = fittedStatistic(colValues)
        }
        
        self.statistics = computedStats
    }
    
    /// Replaces `Double.nan` values with the fitted statistics.
    /// - Parameters:
    ///   - data: Raw input data array or matrix for transformation.
    /// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
    /// - Returns: 2D numerical matrix of shape `[N, P]`.
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
    /// Fits column statistics without materializing nested row arrays.
    /// Missing entries and numeric NaNs are excluded, matching the row API.
    public mutating func fit(_ data: PreparedNumericBatch) throws {
        guard data.rowCount > 0, data.columnCount > 0 else {
            throw PreprocessingError.emptyInput
        }
        statistics = data.columns.map { fittedStatistic($0.values.filter { !$0.isNaN }) }
    }

    /// Fills missing entries and numeric NaNs while preserving the input snapshot.
    public func transform(_ data: PreparedNumericBatch) throws -> PreparedNumericBatch {
        try transform(consuming: data)
    }

    /// Reuses uniquely owned columns when filling missing entries and NaNs.
    /// Shared columns detach only if they need replacement. Clean columns retain
    /// their storage. Output entries are numeric values, including NaN when the
    /// fitted statistic is NaN. Names and original row indices are preserved.
    public func transform(consuming data: consuming PreparedNumericBatch) throws -> PreparedNumericBatch {
        guard let statistics else { throw PreprocessingError.fitNotCalled }
        guard data.rowCount > 0 else {
            return data.replacingNumericColumns(Array(repeating: [], count: data.columnCount))
        }
        guard data.columnCount == statistics.count else {
            throw PreprocessingError.dimensionMismatch(expected: statistics.count, got: data.columnCount)
        }
        var output = consume data
        for column in output.columns.indices {
            if let first = output.columns[column].values.firstIndex(where: { $0.isNaN }) {
                let replacement = statistics[column]
                output.columns[column].values.withUnsafeMutableBufferPointer { values in
                    for row in first..<values.count where values[row].isNaN {
                        values[row] = replacement
                    }
                }
            }
            output.columns[column].validity = nil
            output.columns[column].nullCount = 0
        }
        return output
    }

    /// Fits and transforms prepared columns while preserving their order.
    public mutating func fitTransform(_ data: PreparedNumericBatch) throws -> PreparedNumericBatch {
        try fit(data)
        return try transform(data)
    }

    private func fittedStatistic(_ values: [Double]) -> Double {
        if values.isEmpty {
            if case .constant(let value) = strategy { return value }
            return 0
        }
        switch strategy {
        case .mean:
            return values.reduce(0.0, +) / Double(values.count)
        case .median:
            var sorted = values
            sorted.sort()
            let mid = sorted.count / 2
            return sorted.count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2.0 : sorted[mid]
        case .mostFrequent:
            var counts = [Double: Int]()
            for value in values { counts[value, default: 0] += 1 }
            var best = values[0], maxCount = 0
            for (value, count) in counts {
                if count > maxCount {
                    maxCount = count
                    best = value
                } else if count == maxCount {
                    best = min(best, value)
                }
            }
            return best
        case .constant(let value):
            return value
        }
    }

}
