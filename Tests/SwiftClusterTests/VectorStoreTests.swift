import XCTest
@testable import SwiftCluster

final class VectorStoreTests: XCTestCase {

    func testCosineSimilaritySearch() {
        let store = VectorStore(metric: .cosineSimilarity)
        XCTAssertTrue(store.isEmpty)
        XCTAssertEqual(store.count, 0)

        store.add(id: "doc_exact", vector: [1.0, 0.0, 0.0], metadata: ["topic": "science"])
        store.add(id: "doc_close", vector: [0.9, 0.1, 0.0], metadata: ["topic": "tech"])
        store.add(id: "doc_ortho", vector: [0.0, 1.0, 0.0], metadata: ["topic": "art"])
        store.add(id: "doc_opposite", vector: [-1.0, 0.0, 0.0], metadata: ["topic": "philosophy"])

        XCTAssertEqual(store.count, 4)
        XCTAssertFalse(store.isEmpty)
        XCTAssertTrue(store.contains(id: "doc_exact"))
        XCTAssertFalse(store.contains(id: "non_existent"))

        let results = store.search(query: [1.0, 0.0, 0.0], topK: 3)
        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0].id, "doc_exact")
        XCTAssertEqual(results[0].metadata["topic"], "science")
        XCTAssertEqual(results[0].score, 1.0, accuracy: 1e-6)

        XCTAssertEqual(results[1].id, "doc_close")
        XCTAssertGreaterThan(results[1].score, 0.9)

        XCTAssertEqual(results[2].id, "doc_ortho")
        XCTAssertEqual(results[2].score, 0.0, accuracy: 1e-6)
    }

    func testDotProductSearch() {
        let store = VectorStore(metric: .dotProduct)
        store.add(id: "v1", vector: [2.0, 3.0])
        store.add(id: "v2", vector: [1.0, 1.0])
        store.add(id: "v3", vector: [5.0, 0.0])

        let results = store.search(query: [2.0, 1.0], topK: 2)
        XCTAssertEqual(results.count, 2)
        // v3: 5*2 + 0*1 = 10
        // v1: 2*2 + 3*1 = 7
        // v2: 1*2 + 1*1 = 3
        XCTAssertEqual(results[0].id, "v3")
        XCTAssertEqual(results[0].score, 10.0, accuracy: 1e-6)
        XCTAssertEqual(results[1].id, "v1")
        XCTAssertEqual(results[1].score, 7.0, accuracy: 1e-6)
    }

    func testEuclideanDistanceSearch() {
        let store = VectorStore(metric: .euclideanDistance)
        store.add(id: "pt_origin", vector: [0.0, 0.0])
        store.add(id: "pt_close", vector: [1.0, 1.0])
        store.add(id: "pt_far", vector: [10.0, 10.0])

        let results = store.search(query: [0.0, 0.0], topK: 3)
        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0].id, "pt_origin")
        XCTAssertEqual(results[0].score, 0.0, accuracy: 1e-6)
        XCTAssertEqual(results[1].id, "pt_close")
        XCTAssertEqual(results[1].score, sqrt(2.0), accuracy: 1e-6)
        XCTAssertEqual(results[2].id, "pt_far")
        XCTAssertEqual(results[2].score, sqrt(200.0), accuracy: 1e-6)
    }

    func testBatchInsertAndMutations() {
        let store = VectorStore()
        let batch = [
            VectorEntry(id: "e1", vector: [1.0, 2.0]),
            VectorEntry(id: "e2", vector: [3.0, 4.0]),
            VectorEntry(id: "e3", vector: [5.0, 6.0])
        ]
        store.addBatch(entries: batch)
        XCTAssertEqual(store.count, 3)

        let item = store.get(id: "e2")
        XCTAssertNotNil(item)
        XCTAssertEqual(item?.vector, [3.0, 4.0])

        XCTAssertTrue(store.remove(id: "e2"))
        XCTAssertEqual(store.count, 2)
        XCTAssertNil(store.get(id: "e2"))
        XCTAssertFalse(store.remove(id: "e2"))

        store.clear()
        XCTAssertEqual(store.count, 0)
        XCTAssertTrue(store.isEmpty)
    }

    func testHighDimensionalVectorSearch() {
        let store = VectorStore(metric: .cosineSimilarity)
        let dim = 128
        var targetVector = [Double](repeating: 0.0, count: dim)
        targetVector[0] = 1.0
        targetVector[10] = 2.0
        targetVector[50] = 3.0

        store.add(id: "target", vector: targetVector, metadata: ["type": "dense"])

        for i in 1...20 {
            var noise = [Double](repeating: 0.0, count: dim)
            noise[i] = Double(i) * 0.1
            store.add(id: "noise_\(i)", vector: noise)
        }

        let results = store.search(query: targetVector, topK: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, "target")
        XCTAssertEqual(results[0].score, 1.0, accuracy: 1e-6)
    }
}
