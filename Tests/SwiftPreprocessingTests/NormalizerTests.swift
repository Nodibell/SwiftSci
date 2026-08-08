import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("Normalizer Tests")
struct NormalizerTests {
    
    @Test("Normalizer L2 norm")
    func testL2Normalization() throws {
        var normalizer = Normalizer(norm: .l2)
        let data = [
            [3.0, 4.0],
            [0.0, 0.0]
        ]
        
        let normalized = try normalizer.fitTransform(data)
        
        #expect(normalized[0] == [0.6, 0.8])
        #expect(normalized[1] == [0.0, 0.0])
    }
    
    @Test("Normalizer L1 norm")
    func testL1Normalization() throws {
        var normalizer = Normalizer(norm: .l1)
        let data = [
            [1.0, -3.0]
        ]
        
        let normalized = try normalizer.fitTransform(data)
        #expect(normalized[0] == [0.25, -0.75])
    }
    
    @Test("Normalizer Max norm")
    func testMaxNormalization() throws {
        var normalizer = Normalizer(norm: .max)
        let data = [
            [5.0, -10.0]
        ]
        
        let normalized = try normalizer.fitTransform(data)
        #expect(normalized[0] == [0.5, -1.0])
    }
}
