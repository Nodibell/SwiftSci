import Foundation
import SwiftDataFrame

extension DataFrame {
    private func extractFeatures(columns names: [String]) throws -> [[Double]] {
        let nRows = shape.rows
        guard nRows > 0 else { return [] }
        
        var features = [[Double]](repeating: [Double](repeating: 0.0, count: names.count), count: nRows)
        for (colIdx, colName) in names.enumerated() {
            guard let col = self[column: colName, as: Double.self] else {
                throw DataFrameError.columnNotFound(colName)
            }
            for rowIdx in 0..<nRows {
                features[rowIdx][colIdx] = col[rowIdx] ?? 0.0
            }
        }
        return features
    }

    /// Fits a KMeans clusterer on the specified columns.
    public func fitKMeans(
        columns names: [String],
        k: Int,
        maxIterations: Int = 100,
        tolerance: Double = 1e-4
    ) async throws -> KMeans {
        let features = try extractFeatures(columns: names)
        let kmeans = try KMeans(nClusters: k, maxIterations: maxIterations, tolerance: tolerance)
        try await kmeans.fit(features: features)
        return kmeans
    }
    
    /// Fits a PCA reducer on the specified columns.
    public func fitPCA(
        columns names: [String],
        nComponents: Int
    ) async throws -> PCA {
        let features = try extractFeatures(columns: names)
        let pca = try PCA(nComponents: nComponents)
        try await pca.fit(features)
        return pca
    }
}
