import XCTest
import Accelerate
@testable import SwiftNLP

final class LocalEmbeddingEngineTests: XCTestCase {

    func testEmbeddingDimensionAndUnitNorm() {
        let dim = 128
        let engine = LocalEmbeddingEngine(dimension: dim)

        let embedding = engine.embed("Apple Silicon M3 Max neural network computation")
        XCTAssertEqual(embedding.count, dim)

        // Compute L2 norm
        let sumSq = embedding.reduce(0.0) { $0 + $1 * $1 }
        let norm = sqrt(sumSq)
        XCTAssertEqual(norm, 1.0, accuracy: 1e-4)
    }

    func testDeterministicEmbeddings() {
        let engine = LocalEmbeddingEngine(dimension: 64)
        let text = "Swift 6 Strict Concurrency"

        let v1 = engine.embed(text)
        let v2 = engine.embed(text)

        XCTAssertEqual(v1, v2)
    }

    func testBatchEmbeddings() {
        let engine = LocalEmbeddingEngine(dimension: 32)
        let texts = [
            "Data Science in Swift",
            "High performance computing",
            "Relational database driver"
        ]

        let batch = engine.embedBatch(texts)
        XCTAssertEqual(batch.count, 3)
        for vec in batch {
            XCTAssertEqual(vec.count, 32)
        }
    }

    func testSemanticCosineSimilarity() {
        let engine = LocalEmbeddingEngine(dimension: 128)

        let textML1 = "machine learning neural network deep learning model"
        let textML2 = "deep neural network machine learning optimization"
        let textFruit = "fresh organic sweet banana apple fruit salad"

        let vecML1 = engine.embed(textML1)
        let vecML2 = engine.embed(textML2)
        let vecFruit = engine.embed(textFruit)

        // Dot product between normalized vectors equals cosine similarity
        var dotSim: Double = 0.0
        vDSP_dotprD(vecML1, 1, vecML2, 1, &dotSim, vDSP_Length(128))

        var dotDiff: Double = 0.0
        vDSP_dotprD(vecML1, 1, vecFruit, 1, &dotDiff, vDSP_Length(128))

        XCTAssertGreaterThan(dotSim, dotDiff)
        XCTAssertGreaterThan(dotSim, 0.5)
    }
}
