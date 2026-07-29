import Foundation

/// PolynomialFeatures generates polynomial and interaction features.
public final class PolynomialFeatures: PreprocessingTransformer, @unchecked Sendable {
    /// The degree.
    public let degree: Int
    /// The interaction only.
    public let interactionOnly: Bool
    /// The include bias.
    public let includeBias: Bool
    
    /// The combinations.
    public private(set) var combinations: [[Int]]?
    /// The input feature count.
    public private(set) var inputFeatureCount: Int?
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - degree: The degree.
    ///   - interactionOnly: The interaction only.
    ///   - includeBias: The include bias.
    public init(degree: Int = 2, interactionOnly: Bool = false, includeBias: Bool = true) {
        self.degree = degree
        self.interactionOnly = interactionOnly
        self.includeBias = includeBias
    }
    
    /// Fits the transformer to identify the number of input features and pre-calculate combinations.
    public func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        
        let colCount = data[0].count
        self.inputFeatureCount = colCount
        
        // Generate combinations
        var combos = [[Int]]()
        
        func generate(currentCombo: [Int], lastIdx: Int, currentDegree: Int) {
            if currentDegree > 0 {
                combos.append(currentCombo)
            }
            if currentDegree == degree {
                return
            }
            
            let startIdx = currentDegree == 0 ? 0 : (interactionOnly ? lastIdx + 1 : lastIdx)
            if startIdx >= colCount { return }
            
            for i in startIdx..<colCount {
                generate(currentCombo: currentCombo + [i], lastIdx: i, currentDegree: currentDegree + 1)
            }
        }
        
        generate(currentCombo: [], lastIdx: 0, currentDegree: 0)
        
        // Sort combinations by degree first, then lexicographically
        let sortedCombos = combos.sorted { a, b in
            if a.count != b.count {
                return a.count < b.count
            }
            for (x, y) in zip(a, b) {
                if x != y {
                    return x < y
                }
            }
            return false
        }
        
        self.combinations = sortedCombos
    }
    
    /// Transforms the dataset to include polynomial features.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard let combos = self.combinations, let expectedCols = self.inputFeatureCount else {
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
            
            var polyRow = [Double]()
            if includeBias {
                polyRow.append(1.0)
            }
            
            for combo in combos {
                let product = combo.reduce(1.0) { $0 * row[$1] }
                polyRow.append(product)
            }
            
            transformed.append(polyRow)
        }
        
        return transformed
    }
}
