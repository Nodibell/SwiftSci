import Foundation

/// Distance weight strategy for KNN Imputer.
public enum KNNWeightStrategy: Sendable {
    case uniform
    case distance
}

/// k-Nearest Neighbors missing value imputer for numerical feature matrices.
public struct KNNImputer: PreprocessingTransformer, Sendable {
    /// The n neighbors.
    public let nNeighbors: Int
    /// The weights.
    public let weights: KNNWeightStrategy
    
    private var trainMatrix: [[Double]] = []
    private var columnMeans: [Double] = []
    private var isFitted: Bool = false
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - nNeighbors: The n neighbors.
    ///   - weights: The weights.
    public init(nNeighbors: Int = 5, weights: KNNWeightStrategy = .uniform) {
        self.nNeighbors = max(1, nNeighbors)
        self.weights = weights
    }
    
    /// Fits KNNImputer with feature matrix (PreprocessingTransformer protocol).
    /// - Parameters:
    ///   - data: Raw input data array or matrix for transformation.
    /// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
    public mutating func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        let cols = data[0].count
        for row in data where row.count != cols {
            throw PreprocessingError.dimensionMismatch(expected: cols, got: row.count)
        }
        
        self.trainMatrix = data
        
        // Compute column means for fallback when no valid neighbors exist (Scikit-Learn parity)
        var means = [Double](repeating: 0.0, count: cols)
        for c in 0..<cols {
            var sum = 0.0
            var count = 0
            for r in 0..<data.count {
                let v = data[r][c]
                if !v.isNaN {
                    sum += v
                    count += 1
                }
            }
            means[c] = count > 0 ? (sum / Double(count)) : 0.0
        }
        self.columnMeans = means
        self.isFitted = true
    }
    
    /// Imputes NaN values in feature matrix using weighted k-nearest neighbors (PreprocessingTransformer protocol).
    /// - Parameters:
    ///   - data: Raw input data array or matrix for transformation.
    /// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
    /// - Returns: 2D numerical matrix of shape `[N, P]`.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard isFitted else { throw PreprocessingError.fittingRequired }
        guard !data.isEmpty, !data[0].isEmpty else { throw PreprocessingError.emptyInput }
        
        let expectedCols = trainMatrix[0].count
        for row in data where row.count != expectedCols {
            throw PreprocessingError.dimensionMismatch(expected: expectedCols, got: row.count)
        }
        
        var result = data
        
        for i in 0..<result.count {
            for j in 0..<expectedCols {
                if result[i][j].isNaN {
                    result[i][j] = imputeValue(rowIdx: i, colIdx: j, dataset: data)
                }
            }
        }
        
        return result
    }
    
    private func imputeValue(rowIdx: Int, colIdx: Int, dataset: [[Double]]) -> Double {
        let targetRow = dataset[rowIdx]
        
        // Find candidate neighbors that have non-NaN value at colIdx
        var candidates: [(dist: Double, val: Double)] = []
        
        for k in 0..<trainMatrix.count {
            let neighborRow = trainMatrix[k]
            let val = neighborRow[colIdx]
            if val.isNaN { continue }
            
            // Compute nan-aware Euclidean distance over overlapping non-nan columns
            var distSq = 0.0
            var count = 0
            for c in 0..<targetRow.count {
                if c == colIdx { continue }
                let a = targetRow[c]
                let b = neighborRow[c]
                if !a.isNaN && !b.isNaN {
                    let diff = a - b
                    distSq += diff * diff
                    count += 1
                }
            }
            
            if count > 0 {
                let dist = sqrt(distSq / Double(count) * Double(targetRow.count - 1))
                candidates.append((dist: dist, val: val))
            }
        }
        
        // Fallback to column mean when no neighbors share overlapping valid coordinates
        guard !candidates.isEmpty else {
            return colIdx < columnMeans.count ? columnMeans[colIdx] : 0.0
        }
        
        candidates.sort { $0.dist < $1.dist }
        let topK = Array(candidates.prefix(nNeighbors))
        
        switch weights {
        case .uniform:
            let sum = topK.reduce(0.0) { $0 + $1.val }
            return sum / Double(topK.count)
        case .distance:
            var weightSum = 0.0
            var valSum = 0.0
            for item in topK {
                let w = 1.0 / (item.dist + 1e-5)
                weightSum += w
                valSum += item.val * w
            }
            return weightSum > 0 ? valSum / weightSum : topK[0].val
        }
    }
}
