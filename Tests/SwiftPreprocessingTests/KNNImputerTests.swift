import Testing
@testable import SwiftPreprocessing

@Suite("KNNImputer Tests")
struct KNNImputerTests {
    
    @Test("KNNImputer imputes NaN values accurately using k-nearest neighbors")
    func testKNNImputer() throws {
        let trainData: [[Double]] = [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [10.0, 20.0]
        ]
        
        var imputer = KNNImputer(nNeighbors: 2)
        try imputer.fit(trainData)

        
        let testData: [[Double]] = [
            [1.5, Double.nan]
        ]
        
        let imputed = try imputer.transform(testData)
        
        #expect(!imputed[0][1].isNaN)
        #expect(imputed[0][1] > 2.0 && imputed[0][1] < 5.0)
    }

    @Test("KNNImputer validates matrix rectangularity during fit and transform")
    func testKNNImputerRaggedRowsValidation() throws {
        var imputer = KNNImputer(nNeighbors: 2)
        // Ragged training rows throw dimensionMismatch
        #expect(throws: PreprocessingError.self) {
            try imputer.fit([[1.0, 2.0], [3.0]])
        }

        try imputer.fit([[1.0, 2.0], [3.0, 4.0]])

        // Ragged transform rows throw dimensionMismatch
        #expect(throws: PreprocessingError.self) {
            _ = try imputer.transform([[1.0, 2.0], [5.0]])
        }
        // Column count mismatch throws dimensionMismatch
        #expect(throws: PreprocessingError.self) {
            _ = try imputer.transform([[1.0, 2.0, 3.0]])
        }
    }

    @Test("KNNImputer falls back to training column mean when no neighbors share valid values")
    func testKNNImputerFallbackToColumnMean() throws {
        // Training data with column 0 mean = 100.0, column 1 mean = 200.0
        let trainData: [[Double]] = [
            [90.0, 190.0],
            [110.0, 210.0]
        ]

        var imputer = KNNImputer(nNeighbors: 2)
        try imputer.fit(trainData)

        // Test sample where all coordinate columns are NaN except target column to impute
        // Row has [NaN, NaN] -> no overlapping valid features for distance calculation
        let testData: [[Double]] = [
            [Double.nan, Double.nan]
        ]

        let imputed = try imputer.transform(testData)
        // Both columns should be imputed using their respective column means (100.0 and 200.0)
        #expect(abs(imputed[0][0] - 100.0) < 1e-5)
        #expect(abs(imputed[0][1] - 200.0) < 1e-5)
    }
}
