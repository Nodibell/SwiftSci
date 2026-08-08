import Testing
import Foundation
@testable import SwiftPreprocessing

@Suite("PolynomialFeatures Tests")
struct PolynomialFeaturesTests {
    
    @Test("PolynomialFeatures degree 2 basic")
    func testPolynomialFeaturesBasic() throws {
        var poly = PolynomialFeatures(degree: 2, interactionOnly: false, includeBias: true)
        let data = [
            [2.0, 3.0]
        ]
        
        let transformed = try poly.fitTransform(data)
        
        // Output order should be:
        // Bias, x0, x1, x0^2, x0*x1, x1^2
        // Values: 1.0, 2.0, 3.0, 4.0, 6.0, 9.0
        #expect(transformed[0] == [1.0, 2.0, 3.0, 4.0, 6.0, 9.0])
    }
    
    @Test("PolynomialFeatures degree 2 interaction only without bias")
    func testPolynomialFeaturesInteractionOnlyNoBias() throws {
        var poly = PolynomialFeatures(degree: 2, interactionOnly: true, includeBias: false)
        let data = [
            [2.0, 3.0]
        ]
        
        let transformed = try poly.fitTransform(data)
        
        // Output order should be:
        // x0, x1, x0*x1
        // Values: 2.0, 3.0, 6.0
        #expect(transformed[0] == [2.0, 3.0, 6.0])
    }
}
