import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("OrdinalEncoder Tests")
struct OrdinalEncoderTests {
    
    @Test("OrdinalEncoder basic encoding")
    func testOrdinalEncoderBasic() throws {
        var encoder = OrdinalEncoder()
        let data = [
            ["low", "apple"],
            ["high", "banana"],
            ["medium", "apple"]
        ]
        
        let transformed = try encoder.fitTransform(data)
        
        // Column 0 sorted categories: ["high", "low", "medium"]
        // Column 1 sorted categories: ["apple", "banana"]
        // Expected outputs:
        // "low" -> 1.0, "apple" -> 0.0
        // "high" -> 0.0, "banana" -> 1.0
        // "medium" -> 2.0, "apple" -> 0.0
        
        #expect(transformed[0] == [1.0, 0.0])
        #expect(transformed[1] == [0.0, 1.0])
        #expect(transformed[2] == [2.0, 0.0])
    }
    
    @Test("OrdinalEncoder unknown category fallback")
    func testOrdinalEncoderUnknownFallback() throws {
        var encoder = OrdinalEncoder(unknownValue: -1.0)

        let data = [
            ["apple"],
            ["banana"]
        ]
        
        encoder.fit(data)
        let testData = [
            ["cherry"]
        ]
        
        let transformed = try encoder.transform(testData)
        #expect(transformed[0] == [-1.0])
    }
}
