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
}
