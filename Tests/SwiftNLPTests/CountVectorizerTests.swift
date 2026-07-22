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
}
