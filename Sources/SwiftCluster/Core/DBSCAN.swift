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
        
        let epsSq = eps * eps
        
        func getNeighbors(_ idx: Int) -> [Int] {
            var nbrs = [Int]()
            let pt = features[idx]
            for i in 0..<numSamples {
                guard features[i].count == numFeatures else { continue }
                var distSq = 0.0
                for j in 0..<numFeatures {
                    let diff = pt[j] - features[i][j]
                    distSq += diff * diff
                }
                if distSq <= epsSq {
                    nbrs.append(i)
                }
            }
            return nbrs
        }
        
        var currentClusterId = 0
        
        for i in 0..<numSamples {
            guard features[i].count == numFeatures else {
                throw ClusterError.dimensionMismatch(expected: numFeatures, got: features[i].count)
            }
            
            if visited[i] { continue }
            visited[i] = true
            
            let neighbors = getNeighbors(i)
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
                        let currentNeighbors = getNeighbors(currentPt)
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
