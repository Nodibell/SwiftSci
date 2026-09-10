import Foundation
import Accelerate

/// Comprehensive clustering evaluation metrics and validation indices.
public enum ClusteringMetrics {
    
    /// Computes Inertia (Within-Cluster Sum of Squares, WCSS).
    ///
    /// - Parameters:
    ///   - features: Feature matrix `[numSamples][numFeatures]`.
    ///   - labels: Cluster assignment per sample.
    ///   - centroids: Center coordinates for each cluster `[numClusters][numFeatures]`.
    /// - Returns: Total within-cluster sum of squared distances.
    public static func inertia(features: [[Double]], labels: [Int], centroids: [[Double]]) -> Double {
        var total = 0.0
        for (i, label) in labels.enumerated() {
            guard label >= 0 && label < centroids.count else { continue }
            let distSq = squaredEuclideanDistance(features[i], centroids[label])
            total += distSq
        }
        return total
    }
    
    /// Computes the Calinski-Harabasz Index (Variance Ratio Criterion).
    /// Higher values indicate clusters that are dense and well separated.
    /// - Parameters:
    ///   - features: <#description#>
    ///   - labels: <#description#>
    /// - Returns: <#description#>
    public static func calinskiHarabaszIndex(features: [[Double]], labels: [Int]) -> Double {
        let n = features.count
        guard n > 2 else { return 0.0 }
        
        let uniqueLabels = Array(Set(labels)).filter { $0 >= 0 }
        let k = uniqueLabels.count
        guard k > 1 && k < n else { return 0.0 }
        
        let numFeatures = features[0].count
        var globalMean = [Double](repeating: 0.0, count: numFeatures)
        for row in features {
            for col in 0..<numFeatures {
                globalMean[col] += row[col]
            }
        }
        for col in 0..<numFeatures {
            globalMean[col] /= Double(n)
        }
        
        var extraDispersion = 0.0 // Between-cluster dispersion
        var intraDispersion = 0.0 // Within-cluster dispersion
        
        var clusterMeans = [[Double]](repeating: [Double](repeating: 0.0, count: numFeatures), count: k)
        var clusterCounts = [Int](repeating: 0, count: k)
        let labelToIdx = Dictionary(uniqueKeysWithValues: uniqueLabels.enumerated().map { ($1, $0) })
        
        for (i, label) in labels.enumerated() {
            guard let idx = labelToIdx[label] else { continue }
            clusterCounts[idx] += 1
            for col in 0..<numFeatures {
                clusterMeans[idx][col] += features[i][col]
            }
        }
        
        for idx in 0..<k {
            let count = Double(clusterCounts[idx])
            guard count > 0 else { continue }
            for col in 0..<numFeatures {
                clusterMeans[idx][col] /= count
            }
            extraDispersion += count * squaredEuclideanDistance(clusterMeans[idx], globalMean)
        }
        
        for (i, label) in labels.enumerated() {
            guard let idx = labelToIdx[label] else { continue }
            intraDispersion += squaredEuclideanDistance(features[i], clusterMeans[idx])
        }
        
        guard intraDispersion > 0 else { return 0.0 }
        return (extraDispersion / Double(k - 1)) / (intraDispersion / Double(n - k))
    }
    
    /// Computes the Davies-Bouldin Index.
    /// Lower values indicate better clustering (higher intra-cluster similarity, lower inter-cluster similarity).
    /// - Parameters:
    ///   - features: <#description#>
    ///   - labels: <#description#>
    /// - Returns: <#description#>
    public static func daviesBouldinIndex(features: [[Double]], labels: [Int]) -> Double {
        let n = features.count
        guard n > 2 else { return 0.0 }
        
        let uniqueLabels = Array(Set(labels)).filter { $0 >= 0 }
        let k = uniqueLabels.count
        guard k > 1 else { return 0.0 }
        
        let numFeatures = features[0].count
        let labelToIdx = Dictionary(uniqueKeysWithValues: uniqueLabels.enumerated().map { ($1, $0) })
        
        var clusterMeans = [[Double]](repeating: [Double](repeating: 0.0, count: numFeatures), count: k)
        var clusterCounts = [Int](repeating: 0, count: k)
        
        for (i, label) in labels.enumerated() {
            guard let idx = labelToIdx[label] else { continue }
            clusterCounts[idx] += 1
            for col in 0..<numFeatures {
                clusterMeans[idx][col] += features[i][col]
            }
        }
        
        for idx in 0..<k {
            let count = Double(clusterCounts[idx])
            if count > 0 {
                for col in 0..<numFeatures {
                    clusterMeans[idx][col] /= count
                }
            }
        }
        
        // Calculate average intra-cluster distance S_i for each cluster
        var s = [Double](repeating: 0.0, count: k)
        for (i, label) in labels.enumerated() {
            guard let idx = labelToIdx[label] else { continue }
            let dist = sqrt(squaredEuclideanDistance(features[i], clusterMeans[idx]))
            s[idx] += dist
        }
        for idx in 0..<k {
            let count = Double(clusterCounts[idx])
            s[idx] = count > 0 ? s[idx] / count : 0.0
        }
        
        // Calculate max similarity R_i for each cluster
        var totalDB = 0.0
        for i in 0..<k {
            var maxR = 0.0
            for j in 0..<k where j != i {
                let interDist = sqrt(squaredEuclideanDistance(clusterMeans[i], clusterMeans[j]))
                if interDist > 0 {
                    let r = (s[i] + s[j]) / interDist
                    if r > maxR {
                        maxR = r
                    }
                }
            }
            totalDB += maxR
        }
        
        return totalDB / Double(k)
    }
    
    /// Computes the contamination ratio (percentage of samples flagged as anomalies).
    /// - Parameters:
    ///   - anomalies: <#description#>
    /// - Returns: <#description#>
    public static func contaminationRatio(anomalies: [Bool]) -> Double {
        guard !anomalies.isEmpty else { return 0.0 }
        let count = anomalies.filter { $0 }.count
        return Double(count) / Double(anomalies.count)
    }
    
    /// Computes Adjusted Rand Index (ARI) comparing predicted cluster labels to ground truth.
    /// - Parameters:
    ///   - labelsTrue: <#description#>
    ///   - labelsPred: <#description#>
    /// - Returns: <#description#>
    public static func adjustedRandIndex(labelsTrue: [Int], labelsPred: [Int]) -> Double {
        let n = labelsTrue.count
        guard n > 1 && n == labelsPred.count else { return 0.0 }
        
        var contingency: [Int: [Int: Int]] = [:]
        var a = [Int: Int]()
        var b = [Int: Int]()
        
        for i in 0..<n {
            let t = labelsTrue[i]
            let p = labelsPred[i]
            contingency[t, default: [:]][p, default: 0] += 1
            a[t, default: 0] += 1
            b[p, default: 0] += 1
        }
        
        func choose2(_ n: Int) -> Double {
            return Double(n * (n - 1)) / 2.0
        }
        
        var sumNij = 0.0
        for (_, pMap) in contingency {
            for (_, count) in pMap {
                sumNij += choose2(count)
            }
        }
        
        var sumA = 0.0
        for (_, count) in a {
            sumA += choose2(count)
        }
        
        var sumB = 0.0
        for (_, count) in b {
            sumB += choose2(count)
        }
        
        let sumPairs = choose2(n)
        guard sumPairs > 0 else { return 1.0 }
        
        let expectedIdx = (sumA * sumB) / sumPairs
        let maxIdx = 0.5 * (sumA + sumB)
        let denominator = maxIdx - expectedIdx
        
        if denominator == 0 { return 1.0 }
        return (sumNij - expectedIdx) / denominator
    }
    
    @inline(__always)
    private static func squaredEuclideanDistance(_ a: [Double], _ b: [Double]) -> Double {
        var sumSq = 0.0
        let count = vDSP_Length(a.count)
        vDSP_distancesqD(a, 1, b, 1, &sumSq, count)
        return sumSq
    }
}
