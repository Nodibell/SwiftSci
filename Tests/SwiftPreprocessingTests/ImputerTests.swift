import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("Imputer Tests")
struct ImputerTests {
    
    @Test("Imputer mean strategy")
    func testMeanImputation() throws {
        var imputer = Imputer(strategy: .mean)
        let data = [
            [1.0, 2.0],
            [Double.nan, 4.0],
            [3.0, Double.nan]
        ]
        
        let imputed = try imputer.fitTransform(data)
        #expect(imputer.statistics == [2.0, 3.0])
        
        #expect(imputed[0] == [1.0, 2.0])
        #expect(imputed[1] == [2.0, 4.0])
        #expect(imputed[2] == [3.0, 3.0])
    }
    
    @Test("Imputer median strategy")
    func testMedianImputation() throws {
        var imputer = Imputer(strategy: .median)
        let data = [
            [1.0],
            [10.0],
            [Double.nan],
            [2.0]
        ]
        
        let imputed = try imputer.fitTransform(data)
        #expect(imputer.statistics == [2.0]) // sorted non-nans: 1.0, 2.0, 10.0 -> median is 2.0
        #expect(imputed[2] == [2.0])
    }
    
    @Test("Imputer mostFrequent strategy")
    func testMostFrequentImputation() throws {
        var imputer = Imputer(strategy: .mostFrequent)
        let data = [
            [5.0],
            [3.0],
            [Double.nan],
            [5.0],
            [3.0],
            [3.0]
        ]
        
        let imputed = try imputer.fitTransform(data)
        #expect(imputer.statistics == [3.0])
        #expect(imputed[2] == [3.0])
    }
    
    @Test("Imputer constant strategy")
    func testConstantImputation() throws {
        var imputer = Imputer(strategy: .constant(-999.0))
        let data = [
            [1.0, Double.nan],
            [Double.nan, 2.0]
        ]
        
        let imputed = try imputer.fitTransform(data)
        #expect(imputer.statistics == [-999.0, -999.0])
        #expect(imputed[0] == [1.0, -999.0])
        #expect(imputed[1] == [-999.0, 2.0])
    }

    @Test("Imputer all-NaN column uses 0.0 fallback statistic to ensure valid downstream pipeline imputation (PP-05)")
    func testAllNaNColumn() throws {
        var imputer = Imputer(strategy: .mean)
        let data = [
            [Double.nan],
            [Double.nan]
        ]
        
        let imputed = try imputer.fitTransform(data)
        #expect(imputer.statistics == [0.0])
        #expect(imputed[0] == [0.0])
        #expect(imputed[1] == [0.0])
    }

    @Test("Imputer median does not overflow with large values (PP-05)")
    func testMedianNoOverflow() throws {
        var imputer = Imputer(strategy: .median)
        let data = [
            [Double.greatestFiniteMagnitude * 0.8],
            [Double.greatestFiniteMagnitude * 0.9]
        ]
        
        try imputer.fit(data)
        #expect(imputer.statistics?.count == 1)
        let med = imputer.statistics![0]
        #expect(med.isFinite)
        #expect(abs(med - Double.greatestFiniteMagnitude * 0.85) < 1e10)
    }
}
