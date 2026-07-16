import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("LabelEncoder Tests")
struct LabelEncoderTests {
    
    @Test("LabelEncoder basic encoding and decoding")
    func testLabelEncoderBasic() throws {
        let encoder = LabelEncoder()
        let categories = ["paris", "paris", "tokyo", "amsterdam"]
        
        encoder.fit(categories)
        #expect(encoder.classes == ["amsterdam", "paris", "tokyo"])
        
        let encoded = try encoder.transform(["tokyo", "paris", "amsterdam", "paris"])
        #expect(encoded == [2, 1, 0, 1])
        
        let decoded = try encoder.inverseTransform([2, 1, 0, 1])
        #expect(decoded == ["tokyo", "paris", "amsterdam", "paris"])
    }
    
    @Test("LabelEncoder unknown category throws error")
    func testUnknownCategory() throws {
        let encoder = LabelEncoder()
        encoder.fit(["apple", "banana"])
        
        #expect(throws: PreprocessingError.self) {
            try encoder.transform(["orange"])
        }
        
        #expect(throws: PreprocessingError.self) {
            try encoder.inverseTransform([2])
        }
    }
    
    @Test("LabelEncoder fitNotCalled error")
    func testFitNotCalled() throws {
        let encoder = LabelEncoder()
        
        #expect(throws: PreprocessingError.self) {
            try encoder.transform(["apple"])
        }
        
        #expect(throws: PreprocessingError.self) {
            try encoder.inverseTransform([0])
        }
    }
}
