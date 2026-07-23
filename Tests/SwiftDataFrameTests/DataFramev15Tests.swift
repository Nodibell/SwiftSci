import Testing
import Foundation
@testable import SwiftDataFrame

@Suite("DataFrame v1.5 Features & Optimizations")
struct DataFramev15Tests {
    @Test("DataFrame join inner, left, right, outer")
    func testDataFrameJoin() throws {
        let df1 = try DataFrame(columns: [
            TypedColumn<Int64>(name: "id", values: [1, 2, 3]),
            TypedColumn<String>(name: "name", values: ["Alice", "Bob", "Charlie"])
        ])

        let df2 = try DataFrame(columns: [
            TypedColumn<Int64>(name: "id", values: [2, 3, 4]),
            TypedColumn<Double>(name: "score", values: [95.0, 88.0, 72.0])
        ])

        let inner = try df1.join(df2, on: "id", how: .inner)
        #expect(inner.shape.rows == 2)
        #expect(inner.columnNames.contains("name"))
        #expect(inner.columnNames.contains("score"))

        let left = try df1.join(df2, on: "id", how: .left)
        #expect(left.shape.rows == 3)

        let right = try df1.join(df2, on: "id", how: .right)
        #expect(right.shape.rows == 3)

        let outer = try df1.join(df2, on: "id", how: .outer)
        #expect(outer.shape.rows == 4)
    }

    @Test("DataFrame pivot and melt round-trip")
    func testDataFramePivotMelt() throws {
        let df = try DataFrame(columns: [
            TypedColumn<String>(name: "date", values: ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]),
            TypedColumn<String>(name: "city", values: ["NYC", "LA", "NYC", "LA"]),
            TypedColumn<Double>(name: "temp", values: [32.0, 65.0, 35.0, 68.0])
        ])

        let pivoted = try df.pivot(index: "date", columns: "city", values: "temp")
        #expect(pivoted.shape.rows == 2)
        #expect(pivoted.columnNames.contains("NYC"))
        #expect(pivoted.columnNames.contains("LA"))

        let melted = try pivoted.melt(idVars: ["date"], valueVars: ["NYC", "LA"], varName: "city", valueName: "temp")
        #expect(melted.shape.rows == 4)
    }
}
