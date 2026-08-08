import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("KBinsDiscretizer Tests")
struct KBinsDiscretizerTests {
    
    @Test("KBinsDiscretizer uniform strategy ordinal encode")
    func testUniformOrdinal() throws {
        var kb = KBinsDiscretizer(nBins: 3, strategy: .uniform, encode: .ordinal)
        let data = [
            [0.0],
            [1.5],
            [3.0]
        ]
        
        let transformed = try kb.fitTransform(data)
        
        // Edges: [0.0, 1.0, 2.0, 3.0]
        // 0.0 -> bin 0
        // 1.5 -> bin 1 (1.0 <= val < 2.0)
        // 3.0 -> bin 2
        
        #expect(transformed[0] == [0.0])
        #expect(transformed[1] == [1.0])
        #expect(transformed[2] == [2.0])
    }
    
    @Test("KBinsDiscretizer quantile strategy oneHot encode")
    func testQuantileOneHot() throws {
        var kb = KBinsDiscretizer(nBins: 2, strategy: .quantile, encode: .oneHot)
        let data = [
            [1.0],
            [2.0],
            [3.0],
            [4.0]
        ]
        
        let transformed = try kb.fitTransform(data)
        
        // Median = 2.5. Edges: [1.0, 2.5, 4.0]
        // 1.0, 2.0 -> bin 0 -> [1.0, 0.0]
        // 3.0, 4.0 -> bin 1 -> [0.0, 1.0]
        
        #expect(transformed[0] == [1.0, 0.0])
        #expect(transformed[1] == [1.0, 0.0])
        #expect(transformed[2] == [0.0, 1.0])
        #expect(transformed[3] == [0.0, 1.0])
    }
}
