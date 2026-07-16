import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("MinMaxScaler Tests")
struct MinMaxScalerTests {
    
    @Test("MinMaxScaler basic scaling [0, 1]")
    func testMinMaxScalerBasic() throws {
        let scaler = MinMaxScaler()
        let data: [[Double]] = [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0]
        ]
        
        let scaled = try scaler.fitTransform(data)
        
        #expect(scaler.dataMin == [1.0, 10.0])
        #expect(scaler.dataMax == [3.0, 30.0])
        
        #expect(scaled[0] == [0.0, 0.0])
        #expect(scaled[1] == [0.5, 0.5])
        #expect(scaled[2] == [1.0, 1.0])
    }
    
    @Test("MinMaxScaler custom range [-1, 1]")
    func testMinMaxScalerCustomRange() throws {
        let scaler = MinMaxScaler(range: (-1.0, 1.0))
        let data: [[Double]] = [
            [1.0],
            [2.0],
            [3.0]
        ]
        
        let scaled = try scaler.fitTransform(data)
        
        #expect(scaled[0] == [-1.0])
        #expect(scaled[1] == [0.0])
        #expect(scaled[2] == [1.0])
    }
    
    @Test("MinMaxScaler constant column handling")
    func testMinMaxScalerConstant() throws {
        let scaler = MinMaxScaler()
        let data: [[Double]] = [
            [5.0, 2.0],
            [5.0, 4.0]
        ]
        
        let scaled = try scaler.fitTransform(data)
        
        #expect(scaled[0][0] == 0.0)
        #expect(scaled[1][0] == 0.0)
    }
    
    @Test("MinMaxScaler errors")
    func testErrors() throws {
        let scaler = MinMaxScaler()
        
        #expect(throws: PreprocessingError.self) {
            try scaler.transform([[1.0]])
        }
        
        #expect(throws: PreprocessingError.self) {
            try scaler.fit([])
        }
        
        #expect(throws: PreprocessingError.self) {
            try scaler.fit([[1.0], [1.0, 2.0]])
        }
    }
}
