import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("RobustScaler Tests")
struct RobustScalerTests {
    
    @Test("RobustScaler basic centering and scaling")
    func testRobustScalerBasic() throws {
        var scaler = RobustScaler()
        let data = [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0]
        ]
        
        let scaled = try scaler.fitTransform(data)
        
        // Median = 3.0
        // Q25 = 2.0, Q75 = 4.0 -> IQR = 2.0
        // Scaled values: (val - 3.0) / 2.0
        
        #expect(scaler.center == [3.0])
        #expect(scaler.scale == [2.0])
        
        #expect(scaled[0] == [-1.0])
        #expect(scaled[2] == [0.0])
        #expect(scaled[4] == [1.0])
    }
}
