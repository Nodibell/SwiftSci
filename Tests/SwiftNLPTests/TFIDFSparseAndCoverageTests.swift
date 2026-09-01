import Testing
import Foundation
@testable import SwiftNLP

@Suite("TF-IDF SparseVector & Custom StopWords Coverage Tests")
struct TFIDFSparseAndCoverageTests {

    @Test("SparseVector struct init, toDense, Equatable, and Codable")
    func testSparseVectorLifecycle() throws {
        let sparse = SparseVector(indices: [1, 3, 5], values: [0.5, 0.8, 1.2], dimension: 6)
        #expect(sparse.indices == [1, 3, 5])
        #expect(sparse.values == [0.5, 0.8, 1.2])
        #expect(sparse.dimension == 6)

        // toDense
        let dense = sparse.toDense()
        #expect(dense.count == 6)
        #expect(dense[0] == 0.0)
        #expect(dense[1] == 0.5)
        #expect(dense[2] == 0.0)
        #expect(dense[3] == 0.8)
        #expect(dense[4] == 0.0)
        #expect(dense[5] == 1.2)

        // Equatable
        let sparseCopy = SparseVector(indices: [1, 3, 5], values: [0.5, 0.8, 1.2], dimension: 6)
        #expect(sparse == sparseCopy)

        // Codable JSON roundtrip
        let data = try JSONEncoder().encode(sparse)
        let decoded = try JSONDecoder().decode(SparseVector.self, from: data)
        #expect(decoded == sparse)
    }

    @Test("TFIDFVectorizer transformSparse and fitTransformSparse")
    func testTransformSparse() async throws {
        let corpus = [
            "swift data science machine learning",
            "python machine learning deep learning",
            "swift developer build fast apps"
        ]

        let vectorizer = TFIDFVectorizer()
        let sparseDocs = try await vectorizer.fitTransformSparse(corpus)
        let denseDocs = try await vectorizer.transform(corpus)

        #expect(sparseDocs.count == 3)
        #expect(denseDocs.count == 3)

        for (sparse, dense) in zip(sparseDocs, denseDocs) {
            let reconstructed = sparse.toDense()
            #expect(reconstructed.count == dense.count)
            for (v1, v2) in zip(reconstructed, dense) {
                #expect(abs(v1 - v2) < 1e-9)
            }
        }

        // Test labeled overload
        let sparseLabeled = try await vectorizer.transformSparse(documents: corpus)
        #expect(sparseLabeled.count == 3)
        #expect(sparseLabeled == sparseDocs)
    }

    @Test("TFIDFVectorizer custom stop words and defaultStopWords")
    func testCustomStopWords() async throws {
        #expect(!TFIDFVectorizer.defaultStopWords.isEmpty)
        #expect(TFIDFVectorizer.defaultStopWords.contains("the"))

        // Custom stop words set
        let customStops: Set<String> = ["swift", "test"]
        let vectorizer = TFIDFVectorizer(stopWords: customStops)
        #expect(await vectorizer.stopWords == customStops)

        let corpus = [
            "the swift test library",
            "the data framework"
        ]
        try await vectorizer.fit(corpus)
        let vocab = await vectorizer.vocabulary

        // "swift" and "test" should be removed by custom stop words
        #expect(vocab["swift"] == nil)
        #expect(vocab["test"] == nil)
        // "the" should be kept because it is NOT in custom stop words
        #expect(vocab["the"] != nil)
        #expect(vocab["library"] != nil)
        #expect(vocab["framework"] != nil)
    }

    @Test("TFIDFVectorizer transformSparse error handling and empty input")
    func testTransformSparseErrors() async throws {
        let vectorizer = TFIDFVectorizer()

        // Transform sparse before fitting
        await #expect(throws: NLPError.self) {
            _ = try await vectorizer.transformSparse(["swift science"])
        }

        let corpus = ["apple banana cherry", "banana cherry date"]
        try await vectorizer.fit(corpus)

        // Empty documents array input
        await #expect(throws: NLPError.self) {
            _ = try await vectorizer.transformSparse([])
        }

        // Document with words not in vocabulary or punctuation only
        let res = try await vectorizer.transformSparse(["!@#$%", "unknown words only"])
        #expect(res.count == 2)
        #expect(res[0].indices.isEmpty)
        #expect(res[1].indices.isEmpty)
    }
}
