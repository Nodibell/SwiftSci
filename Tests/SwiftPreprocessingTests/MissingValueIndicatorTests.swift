import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("MissingValueIndicator Tests")
struct MissingValueIndicatorTests {
    
    @Test("MissingValueIndicator missingOnly mode")
    func testMissingOnly() throws {
        var indicator = MissingValueIndicator(features: .missingOnly)
        let data = [
            [1.0, Double.nan],
            [2.0, 3.0]
        ]
        
        let transformed = try indicator.fitTransform(data)
        
        // Only column 1 contains a NaN. So output has 1 column.
        #expect(indicator.indicatorIndices == [1])
        #expect(transformed[0] == [1.0]) // NaN -> 1.0
        #expect(transformed[1] == [0.0]) // 3.0 -> 0.0
    }
    
    @Test("MissingValueIndicator all mode")
    func testAllFeatures() throws {
        var indicator = MissingValueIndicator(features: .all)
        let data = [
            [1.0, Double.nan]
        ]
        
        let transformed = try indicator.fitTransform(data)
        
        #expect(indicator.indicatorIndices == [0, 1])
        #expect(transformed[0] == [0.0, 1.0])
    }
}
