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

    @Test("TargetEncoder OOF encoding prevents target leakage on singleton categories")
    func testTargetEncoderOOFLeakageProtection() throws {
        // Singleton category "Z" with target 1.0; other categories have mean 0.0
        let categories = ["A", "A", "A", "A", "Z"]
        let target = [0.0, 0.0, 0.0, 0.0, 1.0]

        var encoder = TargetEncoder(smoothing: 1.0)
        let oofEncoded = try encoder.fitTransform(categories: categories, target: target, cv: 5)

        // With OOF, fold containing "Z" calculates statistics without "Z" and falls back to out-of-fold global mean (0.2),
        // preventing perfect target leakage (value != 1.0)
        #expect(oofEncoded[4] < 1.0)
        #expect(abs(oofEncoded[4] - 0.0) < 1e-5 || abs(oofEncoded[4] - 0.2) < 1e-5)
    }

    @Test("TargetEncoder validates 2D matrix column count and rectangularity")
    func testTargetEncoder2DMatrixValidation() {
        var encoder = TargetEncoder()
        // Single column throws invalidInput
        #expect(throws: PreprocessingError.self) {
            try encoder.fit([[1.0], [2.0]])
        }
        // Ragged rows throw dimensionMismatch
        #expect(throws: PreprocessingError.self) {
            try encoder.fit([[1.0, 2.0], [3.0]])
        }
    }
}
