import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("PowerTransformer Tests")
struct PowerTransformerTests {
    
    @Test("PowerTransformer Yeo-Johnson transform")
    func testYeoJohnson() throws {
        var pt = PowerTransformer(method: .yeoJohnson, standardize: false)
        let data = [
            [1.0],
            [2.0],
            [3.0]
        ]
        
        let transformed = try pt.fitTransform(data)
        
        #expect(pt.lambdas?.count == 1)
        #expect(transformed.count == 3)
    }
    
    @Test("PowerTransformer Box-Cox positive requirement")
    func testBoxCoxValidation() throws {
        var pt = PowerTransformer(method: .boxCox)
        let data = [
            [0.0],
            [-1.0]
        ]
        
        #expect(throws: PreprocessingError.self) {
            try pt.fit(data)
        }
    }
}
