import Testing
@testable import SwiftPreprocessing

@Suite("TargetEncoder Tests")
struct TargetEncoderTests {
    
    @Test("TargetEncoder fits and transforms categories with smoothing")
    func testTargetEncoder() throws {
        let categories = ["A", "A", "B", "B", "C", "A"]
        let target = [1.0, 1.0, 0.0, 0.0, 0.5, 1.0]
        
        var encoder = TargetEncoder(smoothing: 1.0)
        let encoded = try encoder.fitTransform(categories: categories, target: target)

        
        #expect(encoded.count == 6)
        #expect(encoded[0] > encoded[2]) // Category A (target 1.0) should have higher value than B (target 0.0)
    }
}
