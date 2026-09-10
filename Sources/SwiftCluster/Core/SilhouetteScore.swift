import Foundation
import Accelerate

/// Calculates the Silhouette Coefficient for cluster validation.
///
/// The Silhouette Coefficient measures how similar an object is to its own cluster (cohesion)
/// compared to other clusters (separation). Values range from -1 to +1, where a high value indicates
/// that the object is well matched to its own cluster and poorly matched to neighboring clusters.
public enum SilhouetteScore {
    
    /// Computes the mean Silhouette Coefficient for a dataset given feature matrix and cluster labels.
    ///
    /// - Parameters:
    ///   - features: 2D array of feature values `[numSamples][numFeatures]`.
    ///   - labels: Array of integer cluster assignments for each sample.
    /// - Returns: Mean Silhouette Coefficient across all samples, in range `[-1.0, 1.0]`.
    /// - Throws: <#error description#>
    public static func compute(features: [[Double]], labels: [Int]) throws -> Double {
        let n = features.count
        guard n > 1 else { return 0.0 }
        guard n == labels.count else {
            throw ClusterError.invalidInput("Features count (\(n)) must match labels count (\(labels.count)).")
        }
        
        let uniqueLabels = Array(Set(labels))
        guard uniqueLabels.count > 1 else {
            // Silhouette is undefined for a single cluster
            return 0.0
        }
        
        // Group sample indices by cluster
        var clusterIndices: [Int: [Int]] = [:]
        for (i, label) in labels.enumerated() {
            clusterIndices[label, default: []].append(i)
        }
        
        // Compute pairwise distances matrix
        var distMatrix = [[Double]](repeating: [Double](repeating: 0.0, count: n), count: n)
        for i in 0..<n {
            for j in (i + 1)..<n {
                let d = euclideanDistance(features[i], features[j])
                distMatrix[i][j] = d
                distMatrix[j][i] = d
            }
        }
        
        var totalSilhouette = 0.0
        var validSamples = 0
        
        for i in 0..<n {
            let ownLabel = labels[i]
            let ownCluster = clusterIndices[ownLabel] ?? []
            
            if ownCluster.count <= 1 {
                // Single-element cluster has silhouette of 0
                continue
            }
            
            // a(i): Mean distance to other points in the same cluster
            var sumA = 0.0
            for j in ownCluster where j != i {
                sumA += distMatrix[i][j]
            }
            let a = sumA / Double(ownCluster.count - 1)
            
            // b(i): Min mean distance to points in any other cluster
            var minB = Double.infinity
            for (label, otherIndices) in clusterIndices where label != ownLabel {
                var sumB = 0.0
                for j in otherIndices {
                    sumB += distMatrix[i][j]
                }
                let meanB = sumB / Double(otherIndices.count)
                if meanB < minB {
                    minB = meanB
                }
            }
            
            let b = minB
            let maxAB = max(a, b)
            let s = maxAB > 0 ? (b - a) / maxAB : 0.0
            
            totalSilhouette += s
            validSamples += 1
        }
        
        return validSamples > 0 ? totalSilhouette / Double(validSamples) : 0.0
    }
    
    @inline(__always)
    private static func euclideanDistance(_ a: [Double], _ b: [Double]) -> Double {
        var sumSq = 0.0
        let count = vDSP_Length(a.count)
        vDSP_distancesqD(a, 1, b, 1, &sumSq, count)
        return sqrt(sumSq)
    }
}
