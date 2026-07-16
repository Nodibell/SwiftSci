import Testing
import Foundation
import SwiftDataFrame
@testable import SwiftCluster

@Suite("PCA Tests")
struct PCATests {
    
    @Test("PCA basic fit and transform")
    func testPCABasic() async throws {
        // Synthetic 2D dataset where y is highly correlated with x
        let X: [[Double]] = [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
            [5.0, 10.0]
        ]
        
        let pca = try PCA(nComponents: 1)
        try await pca.fit(X)
        
        let mean = await pca.mean
        #expect(mean != nil)
        #expect(mean!.count == 2)
        #expect(mean![0] == 3.0)
        #expect(mean![1] == 6.0)
        
        let components = await pca.components
        #expect(components != nil)
        #expect(components!.count == 1)
        #expect(components![0].count == 2)
        
        // Principal component should align with the direction of maximum variance [1, 2]
        // Normalized: [1/sqrt(5), 2/sqrt(5)] = [0.4472, 0.8944]
        let pc = components![0]
        #expect(abs(abs(pc[0]) - 0.4472) < 1e-3)
        #expect(abs(abs(pc[1]) - 0.8944) < 1e-3)
        
        let projected = try await pca.transform(X)
        #expect(projected.count == 5)
        #expect(projected[0].count == 1)
        
        // Variance explained should be around 6.25 (since features are [1, 2, 3, 4, 5] and [2, 4, 6, 8, 10])
        let explainedVariance = await pca.explainedVariance
        #expect(explainedVariance != nil)
        #expect(explainedVariance![0] > 0.0)
    }
    
    @Test("PCA components orthogonality")
    func testPCAOrthogonality() async throws {
        let X: [[Double]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
            [10.0, 11.0, 12.0]
        ]
        
        let pca = try PCA(nComponents: 2)
        try await pca.fit(X)
        
        let components = await pca.components
        let pc0 = components![0]
        let pc1 = components![1]
        
        // Dot product of orthogonal components should be zero (or near-zero due to double precision limits)
        let dotProduct = pc0[0]*pc1[0] + pc0[1]*pc1[1] + pc0[2]*pc1[2]
        #expect(abs(dotProduct) < 1e-9)
    }
    
    @Test("PCA error handling")
    func testPCAErrors() async throws {
        // Test invalid nComponents
        #expect(throws: ClusterError.self) {
            _ = try PCA(nComponents: 0)
        }
        
        let pca = try PCA(nComponents: 2)
        
        // Test empty input
        await #expect(throws: ClusterError.self) {
            try await pca.fit([])
        }
        
        // Test fitting required
        await #expect(throws: ClusterError.self) {
            _ = try await pca.transform([[1.0, 2.0]])
        }
        
        // Test fitting with too few samples/features for requested components
        await #expect(throws: ClusterError.self) {
            try await pca.fit([[1.0, 2.0]]) // 1 sample, nComponents is 2
        }
    }

    @Test("KMeans and PCA fitting on DataFrame")
    func testClusterGlue() async throws {
        let f1 = TypedColumn<Double>(name: "f1", values: [1.0, 2.0, 5.0, 6.0])
        let f2 = TypedColumn<Double>(name: "f2", values: [1.0, 1.5, 5.0, 5.5])
        let df = try DataFrame(columns: [f1, f2])

        // KMeans
        let kmeans = try await df.fitKMeans(columns: ["f1", "f2"], k: 2)
        let centroids = await kmeans.getCentroids()
        #expect(centroids != nil)

        // PCA
        let pca = try await df.fitPCA(columns: ["f1", "f2"], nComponents: 1)
        let components = await pca.components
        #expect(components != nil)
        #expect(components!.count == 1)
    }
}
