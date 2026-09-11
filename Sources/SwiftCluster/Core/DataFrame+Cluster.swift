import Foundation
import SwiftDataFrame

extension DataFrame {
    private func extractFeatures(columns names: [String]) throws -> [[Double]] {
        let nRows = shape.rows
        guard nRows > 0 else { return [] }
        
        var features = [[Double]](repeating: [Double](repeating: 0.0, count: names.count), count: nRows)
        for (colIdx, colName) in names.enumerated() {
            guard let col = self[column: colName, as: Double.self] else {
                throw SwiftMLError.columnNotFound(colName)
            }
            for rowIdx in 0..<nRows {
                features[rowIdx][colIdx] = col[rowIdx] ?? 0.0
            }
        }
        return features
    }

    /// Fits a KMeans clusterer on the specified columns.
    /// - Parameters:
    ///   - names: Array of column identifiers or feature names.
    ///   - k: Number of nearest neighbors, clusters, or top components.
    ///   - maxIterations: Maximum number of optimization solver iterations.
    ///   - tolerance: Convergence stopping tolerance threshold.
    /// - Throws: `SwiftMLError` or `ClusterError` if sample count is insufficient or cluster parameters are invalid.
    /// - Returns: Fitted KMeans clustering model.
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
    /// - Parameters:
    ///   - names: Array of column identifiers or feature names.
    ///   - nComponents: Number of latent components or dimensions to retain.
    /// - Throws: `SwiftMLError` or `ClusterError` if sample count is insufficient or cluster parameters are invalid.
    /// - Returns: Fitted `PCA` dimensionality reduction model.
    public func fitPCA(
        columns names: [String],
        nComponents: Int
    ) async throws -> PCA {
        let features = try extractFeatures(columns: names)
        let pca = try PCA(nComponents: nComponents)
        try await pca.fit(features)
        return pca
    }
    
    /// Fits a DBSCAN clusterer on the specified columns.
    /// - Parameters:
    ///   - names: Array of column identifiers or feature names.
    ///   - eps: Maximum neighborhood radius distance epsilon.
    ///   - minSamples: Minimum number of samples required to form a core cluster or split.
    /// - Throws: `SwiftMLError` or `ClusterError` if sample count is insufficient or cluster parameters are invalid.
    /// - Returns: Fitted DBSCAN clustering model.
    public func fitDBSCAN(
        columns names: [String],
        eps: Double = 0.5,
        minSamples: Int = 5
    ) async throws -> DBSCAN {
        let features = try extractFeatures(columns: names)
        let dbscan = DBSCAN(eps: eps, minSamples: minSamples)
        try await dbscan.fit(features: features)
        return dbscan
    }
    
    /// Computes the Silhouette Score for cluster assignments on specified DataFrame columns.
    /// - Parameters:
    ///   - names: Array of column identifiers or feature names.
    ///   - labels: Array of discrete class labels.
    /// - Throws: `SwiftMLError` or `ClusterError` if sample count is insufficient or cluster parameters are invalid.
    /// - Returns: Computed numerical scalar value.
    public func computeSilhouetteScore(columns names: [String], labels: [Int]) throws -> Double {
        let features = try extractFeatures(columns: names)
        return try SilhouetteScore.compute(features: features, labels: labels)
    }
    
    /// Computes Calinski-Harabasz and Davies-Bouldin index metrics for cluster assignments.
    /// - Parameters:
    ///   - names: Array of column identifiers or feature names.
    ///   - labels: Array of discrete class labels.
    /// - Throws: `SwiftMLError` or `ClusterError` if sample count is insufficient or cluster parameters are invalid.
    /// - Returns: The computed (calinskiHarabasz: Double, daviesBouldin: Double) result instance.
    public func computeClusteringMetrics(columns names: [String], labels: [Int]) throws -> (calinskiHarabasz: Double, daviesBouldin: Double) {
        let features = try extractFeatures(columns: names)
        let ch = ClusteringMetrics.calinskiHarabaszIndex(features: features, labels: labels)
        let db = ClusteringMetrics.daviesBouldinIndex(features: features, labels: labels)
        return (ch, db)
    }
}

