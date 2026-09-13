#if os(macOS)
import Testing
import Foundation
@testable import SwiftCluster
import SwiftDataFrame

@Suite("t-SNE Dimensionality Reduction Tests")
struct TSNETests {
    
    @Test("t-SNE reduces dimensionality and preserves sample count")
    func testTSNEBasicProjection() async throws {
        // Create 2 well-separated clusters in 4D
        // Cluster A: around [0, 0, 0, 0]
        // Cluster B: around [10, 10, 10, 10]
        var X: [[Double]] = []
        for i in 0..<8 {
            X.append([Double(i) * 0.1, Double(i) * 0.1, 0.0, 0.1])
        }
        for i in 0..<8 {
            X.append([10.0 + Double(i) * 0.1, 10.0 + Double(i) * 0.1, 10.0, 10.1])
        }
        
        let tsne = TSNE(nComponents: 2, perplexity: 5.0, learningRate: 100.0, maxIterations: 300, seed: 42)
        let embedded = try await tsne.fitTransform(X)
        
        #expect(embedded.count == 16)
        #expect(embedded[0].count == 2)
        
        let kl = await tsne.klDivergence
        #expect(kl != nil)
        #expect(kl! >= 0.0)
    }
    
    @Test("t-SNE reproducibility with fixed seed")
    func testTSNEReproducibility() async throws {
        var X: [[Double]] = []
        for i in 0..<10 {
            X.append([Double(i), Double(i * 2), Double(i * 3)])
        }
        
        let tsne1 = TSNE(nComponents: 2, perplexity: 3.0, maxIterations: 150, seed: 777)
        let tsne2 = TSNE(nComponents: 2, perplexity: 3.0, maxIterations: 150, seed: 777)
        
        let emb1 = try await tsne1.fitTransform(X)
        let emb2 = try await tsne2.fitTransform(X)
        
        for i in 0..<10 {
            #expect(abs(emb1[i][0] - emb2[i][0]) < 1e-6)
            #expect(abs(emb1[i][1] - emb2[i][1]) < 1e-6)
        }
    }
    
    @Test("t-SNE input validation")
    func testTSNEValidation() async {
        let tsne = TSNE(nComponents: 2)
        // Less than 4 samples
        await #expect(throws: (any Error).self) {
            try await tsne.fitTransform([[1.0, 2.0], [3.0, 4.0]])
        }
    }
    
    @Test("DataFrame.tsne generates projection columns")
    func testDataFrameTSNE() async throws {
        var f1 = [Double]()
        var f2 = [Double]()
        var f3 = [Double]()
        
        for i in 0..<12 {
            f1.append(Double(i))
            f2.append(Double(i * 2))
            f3.append(Double(i * 3))
        }
        
        let df = try DataFrame(columns: [
            TypedColumn(name: "f1", values: f1),
            TypedColumn(name: "f2", values: f2),
            TypedColumn(name: "f3", values: f3)
        ])
        
        let projectedDf = try await df.tsne(nComponents: 2, perplexity: 3.0, maxIterations: 150, seed: 42)
        
        #expect(projectedDf.shape.rows == 12)
        #expect(projectedDf.shape.columns == 2)
        #expect(projectedDf.columnNames.contains("tsne_1"))
        #expect(projectedDf.columnNames.contains("tsne_2"))
    }
}
#endif
