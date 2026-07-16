import Testing
import Foundation
@testable import SwiftCluster

@Suite("KMeans Tests")
struct KMeansTests {
    
    @Test("KMeans clustering convergence")
    func testKMeansConvergence() async throws {
        // We have 4 points:
        // Cluster 0: [1.0, 1.0], [1.5, 1.0] (close to center [1, 1])
        // Cluster 1: [10.0, 10.0], [10.5, 10.0] (close to center [10, 10])
        let features: [[Double]] = [
            [1.0, 1.0],
            [1.5, 1.0],
            [10.0, 10.0],
            [10.5, 10.0]
        ]
        
        let kmeans = try KMeans(nClusters: 2)
        try await kmeans.fit(features: features)
        
        let centroids = await kmeans.getCentroids()
        #expect(centroids != nil)
        #expect(centroids!.count == 2)
        #expect(centroids![0].count == 2)
        
        // Predict cluster assignments
        let labels = try await kmeans.predict(features: features)
        #expect(labels.count == 4)
        
        // Point 0 and 1 must belong to the same cluster
        #expect(labels[0] == labels[1])
        // Point 2 and 3 must belong to the same cluster
        #expect(labels[2] == labels[3])
        // Cluster labels for both groups must be different
        #expect(labels[0] != labels[2])
    }
    
    @Test("KMeans error cases")
    func testKMeansErrors() async throws {
        #expect(throws: ClusterError.self) {
            _ = try KMeans(nClusters: 0)
        }
        
        let kmeans = try KMeans(nClusters: 3)
        
        // Empty array
        await #expect(throws: ClusterError.self) {
            try await kmeans.fit(features: [])
        }
        
        // Fitting required
        await #expect(throws: ClusterError.self) {
            _ = try await kmeans.predict(features: [[1.0, 2.0]])
        }
    }
}
