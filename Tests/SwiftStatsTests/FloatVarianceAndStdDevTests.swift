import Testing
import SwiftStats
import SwiftDataFrame
import Foundation

@Suite("Float Variance and Standard Deviation Tests")
struct FloatVarianceAndStdDevTests {

    @Test("Stats.variance with Float array sample and population")
    func testFloatVariance() throws {
        let values: [Float] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]

        // Sample variance (ddof: 1)
        let sampleVar = try Stats.variance(values, ddof: 1)
        #expect(abs(sampleVar - 4.5714285) < 1e-4)

        // Population variance (ddof: 0)
        let popVar = try Stats.variance(values, ddof: 0)
        #expect(abs(popVar - 4.0) < 1e-4)

        // Constant values
        let constant: [Float] = [3.0, 3.0, 3.0, 3.0]
        let constVar = try Stats.variance(constant, ddof: 1)
        #expect(constVar == 0.0)
    }

    @Test("Stats.standardDeviation with Float array")
    func testFloatStandardDeviation() throws {
        let values: [Float] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        let stdDev = try Stats.standardDeviation(values, ddof: 1)
        #expect(abs(stdDev - 2.1380899) < 1e-4)

        let popStdDev = try Stats.standardDeviation(values, ddof: 0)
        #expect(abs(popStdDev - 2.0) < 1e-4)
    }

    @Test("Stats.variance Float error handling")
    func testFloatVarianceErrors() {
        // Empty input
        #expect(throws: SwiftMLError.emptyInput) {
            _ = try Stats.variance([] as [Float], ddof: 1)
        }

        // Invalid DDOF (< 0)
        #expect(throws: SwiftMLError.invalidDDOF(-1)) {
            _ = try Stats.variance([1.0, 2.0] as [Float], ddof: -1)
        }

        // Insufficient data (n <= ddof)
        #expect(throws: SwiftMLError.insufficientData(minimum: 3, got: 2)) {
            _ = try Stats.variance([1.0, 2.0] as [Float], ddof: 2)
        }
    }
}
