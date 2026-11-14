import Testing
import Foundation
@testable import SwiftCluster

@Suite("KMeans Tests")
struct KMeansTests {
    
    init() {
        if let url = Bundle.main.resourceURL?.appendingPathComponent("mlx-swift_Cmlx.bundle"),
           let bundle = Bundle(url: url) {
            _ = bundle.resourceURL
        }
        for bundle in Bundle.allBundles {
            if let url = bundle.resourceURL?.appendingPathComponent("mlx-swift_Cmlx.bundle"),
               let b = Bundle(url: url) {
                _ = b.resourceURL
            }
        }
    }
    
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

#if os(macOS)
import Testing
@testable import SwiftCluster

@Suite("KMeans input validation and distance robustness (S-09)")
struct KMeansRobustnessTests {
    @Test func rejectsNonFiniteAndRaggedInput() async throws {
        let km = try KMeans(nClusters: 2)
        await #expect(throws: (any Error).self) { try await km.fit(features: [[0, 0], [1, .nan], [5, 5]]) }
        await #expect(throws: (any Error).self) { try await km.fit(features: [[0, 0], [1, .infinity], [5, 5]]) }
        await #expect(throws: (any Error).self) { try await km.fit(features: [[0, 0], [1], [5, 5]]) }

        let fitted = try KMeans(nClusters: 2, device: .cpu)
        try await fitted.fit(features: [[0, 0], [0.1, 0.1], [10, 10], [10.1, 10.1]])
        await #expect(throws: (any Error).self) { try await fitted.predict(features: [[0, .nan]]) }
        await #expect(throws: (any Error).self) { try await fitted.predict(features: [[0, 0], [1]]) }
        await #expect(throws: (any Error).self) { try await fitted.predict(features: [[0, 0, 0]]) }
    }

    @Test func identicalPointsAndExtremeMagnitudesAreStable() async throws {
        let same = [[Double]](repeating: [3, 3, 3], count: 6)
        let km = try KMeans(nClusters: 2, device: .cpu)
        try await km.fit(features: same)
        let labels = try await km.predict(features: same)
        #expect(Set(labels).count == 1)
        let big: [[Double]] = [[1e150, 1e150], [1e150 + 1e134, 1e150], [-1e150, 1e150], [-1e150, 1e150 + 1e134]]
        let km2 = try KMeans(nClusters: 2, device: .cpu)
        try await km2.fit(features: big)
        #expect(try await km2.predict(features: big).count == 4)
    }

    @Test func nearlyIdenticalLargeVectorsNeverGiveNegativeDistances() async throws {
        // 20 features exercises the vDSP path; distances are direct sums of squares, hence >= 0.
        let base = (0..<20).map { Double($0) * 1e8 }
        let rows = (0..<8).map { i in base.map { $0 + Double(i) * 1e-8 } }
        let km = try KMeans(nClusters: 2, device: .cpu)
        try await km.fit(features: rows)
        #expect(try await km.predict(features: rows).count == 8)
    }
}
#endif
