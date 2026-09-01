import Foundation
import SwiftPreprocessing

/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering on CPU.
public actor DBSCAN {
    /// The eps.
    public let eps: Double
    /// The min samples.
    public let minSamples: Int
    /// The requested device.
    public let requestedDevice: ExecutionDevice
    
    /// The resolved device.
    public private(set) var resolvedDevice: ExecutionDevice?
    /// The labels.
    public private(set) var labels: [Int]?
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - eps: The eps.
    ///   - minSamples: The min samples.
    ///   - device: The device.
    public init(eps: Double = 0.5, minSamples: Int = 5, device: ExecutionDevice = .cpu) {
        self.eps = eps
        self.minSamples = minSamples
        self.requestedDevice = device
    }
    
    /// Fits the DBSCAN model on the input dataset.
    public func fit(features: [[Double]]) async throws {
        guard !features.isEmpty else {
            throw ClusterError.emptyInput
        }
        
        let numSamples = features.count
        let numFeatures = features[0].count
        
        if requestedDevice == .gpu {
            print("Warning: DBSCAN does not support GPU. Falling back to CPU execution.")
        }
        self.resolvedDevice = .cpu
        
        var labelsLocal = [Int](repeating: -1, count: numSamples) // -1 denotes noise/unassigned
        var visited = [Bool](repeating: false, count: numSamples)
        
        // Build KD-Tree for O(N log N) range neighborhood search
        let tree = KDTree(points: features)
        
        var currentClusterId = 0
        
        for i in 0..<numSamples {
            guard features[i].count == numFeatures else {
                throw ClusterError.dimensionMismatch(expected: numFeatures, got: features[i].count)
            }
            
            if visited[i] { continue }
            visited[i] = true
            
            let neighbors = tree.queryRadius(point: features[i], radius: eps)
            if neighbors.count < minSamples {
                labelsLocal[i] = -1 // Noise
            } else {
                labelsLocal[i] = currentClusterId
                
                var queue = neighbors
                var queueIdx = 0
                
                // Track items in queue to avoid infinite loops/duplicates
                var inQueue = Set(neighbors)
                
                while queueIdx < queue.count {
                    let currentPt = queue[queueIdx]
                    queueIdx += 1
                    
                    if !visited[currentPt] {
                        visited[currentPt] = true
                        let currentNeighbors = tree.queryRadius(point: features[currentPt], radius: eps)
                        if currentNeighbors.count >= minSamples {
                            for neighbor in currentNeighbors {
                                if !inQueue.contains(neighbor) {
                                    inQueue.insert(neighbor)
                                    queue.append(neighbor)
                                }
                            }
                        }
                    }
                    
                    if labelsLocal[currentPt] == -1 {
                        labelsLocal[currentPt] = currentClusterId
                    }
                }
                
                currentClusterId += 1
            }
        }
        
        self.labels = labelsLocal
    }
    
    /// Fits DBSCAN and returns the labels.
    public func fitTransform(features: [[Double]]) async throws -> [Int] {
        try await fit(features: features)
        return labels ?? []
    }
}
