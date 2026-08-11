import Testing
@testable import SwiftPreprocessing

@Suite("FrequencyEncoder Tests")
struct FrequencyEncoderTests {
    
    @Test("FrequencyEncoder calculates correct frequency ratios")
    func testFrequencyEncoder() throws {
        let categories = ["Apple", "Apple", "Apple", "Banana", "Cherry"]
        var encoder = FrequencyEncoder()
        let encoded = try encoder.fitTransform(categories: categories)

        
        #expect(encoded[0] == 0.6) // 3 / 5
        #expect(encoded[3] == 0.2) // 1 / 5
        #expect(encoded[4] == 0.2) // 1 / 5
    }
}
