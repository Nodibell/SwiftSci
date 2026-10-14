import Testing
import Foundation
@testable import SwiftNLP

@Suite("CountVectorizer Tests")
struct CountVectorizerTests {
    
    @Test("CountVectorizer builds vocabulary and transforms documents")
    func testCountVectorizer() {
        let docs = [
            "apple banana apple",
            "banana cherry"
        ]
        let vec = CountVectorizer(lowercase: true)
        let matrix = vec.fitTransform(documents: docs)
        
        #expect(matrix.count == 2)
        #expect(matrix[0].count == 3) // apple, banana, cherry
        #expect(matrix[0] == [2.0, 1.0, 0.0]) // apple:2, banana:1, cherry:0
        #expect(matrix[1] == [0.0, 1.0, 1.0]) // apple:0, banana:1, cherry:1
    }

    @Test("CountVectorizer deterministically breaks ties by alphabetical order")
    func testCountVectorizerDeterministicTieBreaking() {
        let docs = [
            "zebra apple",
            "zebra apple"
        ]

        let vec = CountVectorizer(maxFeatures: 1, lowercase: true)
        let matrix = vec.fitTransform(documents: docs)

        #expect(matrix.count == 2)
        #expect(matrix[0].count == 1)
        #expect(vec.vocabulary.keys.contains("apple"))
        #expect(!vec.vocabulary.keys.contains("zebra"))
    }

    @Test("CountVectorizer concurrent read/transform safety")
    func testCountVectorizerThreadSafety() async throws {
        let corpus = [
            "swift data science machine learning text processing",
            "natural language processing with accelerate framework",
            "naive bayes classification and text vectorization"
        ]

        let countVec = CountVectorizer()
        _ = countVec.fitTransform(documents: corpus)

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    _ = countVec.transform(documents: ["swift machine learning"])
                    _ = countVec.vocabulary
                }
            }
        }
    }
}

