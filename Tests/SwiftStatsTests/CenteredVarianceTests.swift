import Testing
import SwiftStats
import SwiftDataFrame

@Suite("Centered variance precision")
struct CenteredVarianceTests {
    @Test("Preserves a mean between adjacent large Double values")
    func unrepresentableAbsoluteMean() throws {
        let values = [1e15 + 5.75, 1e15 - 5.875]
        #expect(try Stats.variance(values) == 67.5703125)
        #expect(try Stats.variance(values, ddof: 0) == 33.78515625)
        #expect(try Stats.variance(Array(values.reversed())) == 67.5703125)
    }

    @Test("Block boundaries preserve translated sample and population variance",
          arguments: [1, 3, 51, 52, 819])
    func translatedBlocks(repetitions: Int) throws {
        let values = (0..<(5 * repetitions)).map { 1e12 + Double($0 % 5 - 2) }
        let population = try Stats.variance(values, ddof: 0)
        let sample = try Stats.variance(values)
        let expected = 2 * Double(values.count) / Double(values.count - 1)
        #expect(abs(population - 2) < 1e-14)
        #expect(abs(sample - expected) < 1e-14)
    }

    @Test("Singleton population and constant samples have zero variance")
    func constant() throws {
        #expect(try Stats.variance([1e15], ddof: 0) == 0)
        #expect(try Stats.variance(Array(repeating: 1e15, count: 513)) == 0)
    }

    @Test("Retains representable small variance")
    func smallMagnitude() throws {
        let result = try Stats.variance([1e-150, -1e-150])
        #expect(abs(result / 2e-300 - 1) < 1e-14)
    }

    @Test("Overflowing centered differences retain the existing infinite variance")
    func overflowingRange() throws {
        #expect(try Stats.variance([-1e308, 1e308]) == .infinity)
    }

    @Test("Input validation survives the optimized path")
    func validation() throws {
        #expect(throws: StatsError.containsNaN) { try Stats.variance([0.0, Double.nan]) }
        #expect(throws: StatsError.emptyInput) { try Stats.variance([] as [Double]) }
        #expect(throws: StatsError.invalidDDOF(-1)) { try Stats.variance([1.0, 2.0] as [Double], ddof: -1) }
    }
}
