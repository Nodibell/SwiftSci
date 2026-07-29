import Testing
@testable import SwiftDataFrame

@Suite("DataFrame SIMD Bitmask Filter Tests (Phase 1)")
struct DataFrameSIMDFilterTests {

    @Test("SIMD Double comparison filtering produces exact matching results")
    func testSIMDDoubleFiltering() throws {
        let values: [Double] = (0..<1000).map { Double($0) * 0.5 }
        let df = try DataFrame(columns: [
            TypedColumn<Double>(name: "value", values: values.map { $0 })
        ])

        let filteredGt = try df.filter(column: "value", where: .greaterThan(250.0))
        #expect(filteredGt.rowCount == 499)
        #expect((filteredGt[column: "value"] as? TypedColumn<Double>)?.values.first == 250.5)

        let filteredLt = try df.filter(column: "value", where: .lessThan(50.0))
        #expect(filteredLt.rowCount == 100)
        #expect((filteredLt[column: "value"] as? TypedColumn<Double>)?.values.last == 49.5)

        let filteredGte = try df.filter(column: "value", where: .greaterThanOrEqual(250.0))
        #expect(filteredGte.rowCount == 500)
        #expect((filteredGte[column: "value"] as? TypedColumn<Double>)?.values.first == 250.0)

        let filteredLte = try df.filter(column: "value", where: .lessThanOrEqual(50.0))
        #expect(filteredLte.rowCount == 101)

        let filteredEq = try df.filter(column: "value", where: .equals(100.0))
        #expect(filteredEq.rowCount == 1)
        #expect((filteredEq[column: "value"] as? TypedColumn<Double>)?.values.first == 100.0)

        let filteredNeq = try df.filter(column: "value", where: .notEquals(100.0))
        #expect(filteredNeq.rowCount == 999)
    }

    @Test("SIMD Int64 comparison filtering produces exact matching results")
    func testSIMDInt64Filtering() throws {
        let values: [Int64] = (0..<1000).map { Int64($0) }
        let df = try DataFrame(columns: [
            TypedColumn<Int64>(name: "id", values: values.map { $0 })
        ])

        let filteredGt = try df.filter(column: "id", where: .greaterThan(500))
        #expect(filteredGt.rowCount == 499)

        let filteredLte = try df.filter(column: "id", where: .lessThanOrEqual(500))
        #expect(filteredLte.rowCount == 501)

        let filteredEq = try df.filter(column: "id", where: .equals(777))
        #expect(filteredEq.rowCount == 1)
        #expect((filteredEq[column: "id"] as? TypedColumn<Int64>)?.values.first == 777)
    }

    @Test("SIMD filtering handles empty dataframes and boundary limits")
    func testSIMDEmptyAndLimits() throws {
        let emptyDf = DataFrame.empty
        #expect(emptyDf.rowCount == 0)

        let smallValues: [Double] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let df = try DataFrame(columns: [
            TypedColumn<Double>(name: "x", values: smallValues.map { $0 })
        ])

        let filtered = try df.filter(column: "x", where: .greaterThan(10.0))
        #expect(filtered.rowCount == 0)
    }
}
