import Foundation
import MLX

/// K-Means Clustering algorithm using GPU-accelerated MLX operators.
public actor KMeans {
    /// Number of clusters.
    public let nClusters: Int
    
    /// Maximum number of iterations.
    public let maxIterations: Int
    
    /// Convergence tolerance.
    public let tolerance: Float
    
    /// Centroids of clusters. Shape: [nClusters, nFeatures]
    public private(set) var centroids: MLXArray?
    
    /// Initializes K-Means with parameters.
    public init(nClusters: Int, maxIterations: Int = 300, tolerance: Double = 1e-4) throws {
        guard nClusters > 0 else {
            throw ClusterError.invalidParameter("nClusters must be greater than 0.")
        }
        guard maxIterations > 0 else {
            throw ClusterError.invalidParameter("maxIterations must be greater than 0.")
        }
        self.nClusters = nClusters
        self.maxIterations = maxIterations
        self.tolerance = Float(tolerance)
    }
    
    /// Fits K-Means on the input dataset (Sendable interface).
    public func fit(features: [[Double]]) throws {
        guard !features.isEmpty else {
            throw ClusterError.emptyInput
        }
        
        let numSamples = features.count
        let numFeatures = features[0].count
        
        let X = MLXArray(features.flatMap { $0.map { Float($0) } }).reshaped([numSamples, numFeatures])
        try fit(X: X)
    }
    
    /// Fits K-Means on the input dataset X.
    /// - Parameter X: A 2D MLXArray of shape [samples, features].
    public func fit(X: MLXArray) throws {
        guard X.size > 0 else {
            throw ClusterError.emptyInput
        }
        
        let shape = X.shape
        guard shape.count == 2 else {
            throw ClusterError.dimensionMismatch(expected: 2, got: shape.count)
        }
        
        let numSamples = shape[0]
        
        guard nClusters <= numSamples else {
            throw ClusterError.invalidParameter("nClusters (\(nClusters)) cannot be greater than the number of samples (\(numSamples)).")
        }
        
        // 1. Initialize centroids (using the first nClusters samples)
        var currentCentroids = X[0..<nClusters]
        
        for _ in 0..<maxIterations {
            // 2. Compute pairwise distances: shape [samples, nClusters]
            // X: [N, 1, M]
            // Centroids: [1, K, M]
            let diff = X.expandedDimensions(axes: [1]) - currentCentroids.expandedDimensions(axes: [0])
            let dists = sqrt((diff * diff).sum(axis: -1))
            
            // 3. Assign labels (closest centroid): shape [samples]
            let labels = argMin(dists, axis: -1)
            
            // 4. Compute new centroids
            var updatedCentroids = [MLXArray]()
            for k in 0..<nClusters {
                let mask = equal(labels, MLXArray(k))
                let count = mask.sum().item(Int.self)
                
                if count > 0 {
                    // Sum features for samples belonging to cluster k
                    let sumPoints = (X * mask.expandedDimensions(axes: [1])).sum(axis: 0)
                    let newCentroid = sumPoints / Float(count)
                    updatedCentroids.append(newCentroid)
                } else {
                    // Keep existing centroid if no points are assigned to it
                    updatedCentroids.append(currentCentroids[k])
                }
            }
            
            let newCentroids = stacked(updatedCentroids)
            
            // 5. Check for convergence (distance between old and new centroids)
            let diffCentroids = newCentroids - currentCentroids
            let distChange = sqrt((diffCentroids * diffCentroids).sum()).item(Float.self)
            
            currentCentroids = newCentroids
            
            // Eagerly evaluate centroids to free memory and compile graph steps
            eval(currentCentroids)
            
            if distChange < tolerance {
                break
            }
        }
        
        self.centroids = currentCentroids
    }
    
    /// Assigns each sample to the closest centroid (Sendable interface).
    public func predict(features: [[Double]]) throws -> [Int] {
        guard !features.isEmpty else {
            return []
        }
        
        let numSamples = features.count
        let numFeatures = features[0].count
        
        let X = MLXArray(features.flatMap { $0.map { Float($0) } }).reshaped([numSamples, numFeatures])
        let labels = try predict(X: X)
        
        return labels.asArray(Int32.self).map { Int($0) }
    }
    
    /// Assigns each sample in X to the closest centroid.
    /// - Parameter X: A 2D MLXArray of shape [samples, features].
    /// - Returns: An MLXArray of shape [samples] containing cluster indices.
    public func predict(X: MLXArray) throws -> MLXArray {
        guard let centroids = self.centroids else {
            throw ClusterError.fittingRequired
        }
        guard X.size > 0 else {
            throw ClusterError.emptyInput
        }
        
        let shape = X.shape
        guard shape.count == 2 else {
            throw ClusterError.dimensionMismatch(expected: 2, got: shape.count)
        }
        
        let numFeatures = centroids.shape[1]
        guard shape[1] == numFeatures else {
            throw ClusterError.dimensionMismatch(expected: numFeatures, got: shape[1])
        }
        
        let diff = X.expandedDimensions(axes: [1]) - centroids.expandedDimensions(axes: [0])
        let dists = sqrt((diff * diff).sum(axis: -1))
        let labels = argMin(dists, axis: -1)
        
        return labels
    }
    
    /// Returns the learned centroids as a standard Sendable 2D Double array.
    public func getCentroids() -> [[Double]]? {
        guard let centroids = centroids else { return nil }
        let flatArray = centroids.asArray(Float.self)
        let numClusters = centroids.shape[0]
        let numFeatures = centroids.shape[1]
        
        var result = [[Double]]()
        for i in 0..<numClusters {
            var row = [Double]()
            for j in 0..<numFeatures {
                row.append(Double(flatArray[i * numFeatures + j]))
            }
            result.append(row)
        }
        return result
    }
}
