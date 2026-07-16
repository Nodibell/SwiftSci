import Foundation

/// Encodes categorical features into one-hot binary matrices.
public final class OneHotEncoder: @unchecked Sendable {
    /// Categorical levels for each feature column.
    public private(set) var categories: [[String]] = []
    
    public init() {}
    
    /// Fits the OneHotEncoder to a 2D categorical dataset of shape [rows, cols].
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
    
    /// Transforms the categorical dataset into a one-hot encoded matrix.
    /// - Parameter data: A 2D string dataset of shape [rows, cols].
    /// - Returns: A 2D array of doubles with concatenated one-hot vectors.
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
        
        // Build lookups for fast index retrieval
        let lookups = categories.map { categoryList -> Dictionary<String, Int> in
            Dictionary(uniqueKeysWithValues: categoryList.enumerated().map { ($0.element, $0.offset) })
        }
        
        // Pre-calculate target vector length
        let targetLength = categories.reduce(0) { $0 + $1.count }
        
        for row in data {
            guard row.count == colCount else {
                throw PreprocessingError.dimensionMismatch(expected: colCount, got: row.count)
            }
            
            var oneHotRow = [Double]()
            oneHotRow.reserveCapacity(targetLength)
            
            for col in 0..<colCount {
                let category = row[col]
                let categoryList = categories[col]
                let lookup = lookups[col]
                
                guard let idx = lookup[category] else {
                    throw PreprocessingError.unknownCategory(category)
                }
                
                // Append one-hot vector for the current column
                for i in 0..<categoryList.count {
                    oneHotRow.append(i == idx ? 1.0 : 0.0)
                }
            }
            transformed.append(oneHotRow)
        }
        
        return transformed
    }
    
    /// Fits to categorical data, then transforms it.
    public func fitTransform(_ data: [[String]]) throws -> [[Double]] {
        fit(data)
        return try transform(data)
    }
}
