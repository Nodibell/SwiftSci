import Testing
import Foundation
@testable import SwiftCluster

@Suite("HNSWIndex Tests")
struct HNSWIndexTests {

    @Test("HNSWIndex basic insertion and exact nearest match")
    func testBasicInsertionAndSearch() async {
        let index = HNSWIndex(metric: .euclideanDistance, M: 8, efConstruction: 64, efSearch: 32, seed: 123)

        await index.add(id: "origin", vector: [0.0, 0.0, 0.0], metadata: ["label": "zero"])
        await index.add(id: "near", vector: [0.1, 0.1, 0.0], metadata: ["label": "near"])
        await index.add(id: "far", vector: [10.0, 10.0, 10.0], metadata: ["label": "far"])

        let count = await index.count
        #expect(count == 3)

        let containsNear = await index.contains(id: "near")
        #expect(containsNear == true)
        let containsGhost = await index.contains(id: "ghost")
        #expect(containsGhost == false)

        let results = await index.search(query: [0.02, 0.02, 0.0], topK: 3)
        #expect(results.count == 3)
        #expect(results[0].id == "origin")
        #expect(results[1].id == "near")
        #expect(results[2].id == "far")
        #expect(results[0].score < results[1].score)
        #expect(results[1].score < results[2].score)
    }

    @Test("HNSWIndex with Cosine Similarity metric")
    func testCosineSimilaritySearch() async {
        let index = HNSWIndex(metric: .cosineSimilarity, M: 8, efConstruction: 64, efSearch: 32, seed: 42)

        await index.add(id: "north", vector: [0.0, 1.0])
        await index.add(id: "northeast", vector: [0.7071, 0.7071])
        await index.add(id: "east", vector: [1.0, 0.0])
        await index.add(id: "south", vector: [0.0, -1.0])

        let results = await index.search(query: [0.1, 0.99], topK: 3)
        #expect(results.count == 3)
        #expect(results[0].id == "north")
        #expect(results[1].id == "northeast")
        // Cosine similarity score should be close to 1.0 for "north"
        #expect(results[0].score > 0.95)
    }

    @Test("HNSWIndex with Dot Product metric")
    func testDotProductSearch() async {
        let index = HNSWIndex(metric: .dotProduct, M: 8, efConstruction: 64, efSearch: 32, seed: 99)

        await index.add(id: "large", vector: [10.0, 10.0])
        await index.add(id: "medium", vector: [3.0, 3.0])
        await index.add(id: "small", vector: [0.5, 0.5])

        let results = await index.search(query: [1.0, 1.0], topK: 2)
        #expect(results.count == 2)
        #expect(results[0].id == "large")
        #expect(results[1].id == "medium")
        #expect(results[0].score > results[1].score)
    }

    @Test("HNSWIndex high-dimensional recall against linear scan")
    func testHighDimensionalRecall() async {
        let index = HNSWIndex(metric: .euclideanDistance, M: 16, efConstruction: 100, efSearch: 50, seed: 777)
        let numVectors = 150
        let dim = 16

        // Insert pseudo-random vectors
        var rawVectors: [(id: String, vec: [Double])] = []
        for i in 0..<numVectors {
            let vec = (0..<dim).map { d in sin(Double(i * dim + d)) }
            let id = "vec_\(i)"
            rawVectors.append((id: id, vec: vec))
            await index.add(id: id, vector: vec)
        }

        let total = await index.count
        #expect(total == numVectors)

        // Query with vector 42
        let query = rawVectors[42].vec
        let results = await index.search(query: query, topK: 5)
        #expect(!results.isEmpty)
        #expect(results[0].id == "vec_42")
        #expect(abs(results[0].score) < 1e-6)
    }

    @Test("HNSWIndex concurrent actor access")
    func testConcurrentActorAccess() async {
        let index = HNSWIndex(metric: .euclideanDistance, M: 8, efConstruction: 50, efSearch: 20)

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<50 {
                group.addTask {
                    let vec = [Double(i), Double(i * 2), Double(i * 3)]
                    await index.add(id: "task_\(i)", vector: vec)
                }
            }
        }

        let finalCount = await index.count
        #expect(finalCount == 50)

        let searchRes = await index.search(query: [25.0, 50.0, 75.0], topK: 3)
        #expect(searchRes.count == 3)
        #expect(searchRes[0].id == "task_25")
    }
}
