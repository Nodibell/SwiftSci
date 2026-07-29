import Foundation

/// OrdinalEncoder encodes 2D categorical features (strings) as integer ordinals (Double).
public final class OrdinalEncoder: @unchecked Sendable {
    /// The categories.
    public private(set) var categories: [[String]] = []
    /// The unknown value.
    public let unknownValue: Double?
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - unknownValue: The unknown value.
    public init(unknownValue: Double? = nil) {
        self.unknownValue = unknownValue
    }
    
    /// Fits the OrdinalEncoder by identifying unique sorted categories for each column.
    public func fit(_ data: [[String]]) {
        guard !data.isEmpty, !data[0].isEmpty else {
            self.categories = []
            return
        }
        
        let colCount = data[0].count
        var uniques = [Set<String>](repeating: Set<String>(), count: colCount)
        
        for row in data {
            for col in 0..<min(colCount, row.count) {
                uniques[col].insert(row[col])
            }
        }
        
        self.categories = uniques.map { $0.sorted() }
    }
    
    /// Transforms the categorical dataset into ordinal values.
    public func transform(_ data: [[String]]) throws -> [[Double]] {
        guard !categories.isEmpty else {
            throw PreprocessingError.fitNotCalled
        }
        guard !data.isEmpty else {
            return []
        }
        
        let colCount = categories.count
        var transformed = [[Double]]()
        transformed.reserveCapacity(data.count)
        
        // Build lookups for fast retrieval
        let lookups = categories.map { categoryList -> Dictionary<String, Double> in
            Dictionary(uniqueKeysWithValues: categoryList.enumerated().map { ($0.element, Double($0.offset)) })
        }
        
        for row in data {
            guard row.count == colCount else {
                throw PreprocessingError.dimensionMismatch(expected: colCount, got: row.count)
            }
            
            let ordinalRow = try (0..<colCount).map { col -> Double in
                let category = row[col]
                let lookup = lookups[col]
                
                if let val = lookup[category] {
                    return val
                } else if let fallback = unknownValue {
                    return fallback
                } else {
                    throw PreprocessingError.unknownCategory(category)
                }
            }
            transformed.append(ordinalRow)
        }
        
        return transformed
    }
    
    /// Fits to categorical data, then transforms it.
    public func fitTransform(_ data: [[String]]) throws -> [[Double]] {
        fit(data)
        return try transform(data)
    }
}
