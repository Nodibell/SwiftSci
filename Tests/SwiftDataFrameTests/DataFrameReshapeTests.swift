import Testing
import Foundation
@testable import SwiftDataFrame

@Suite("DataFrame Join & Reshape Tests")
struct DataFrameReshapeTests {
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

    @Test("Pivot distinguishes literal 'null' string from missing nil and preserves index dtype")
    func testPivotDistinguishesNullStringFromMissing() throws {
        let df = try DataFrame(columns: [
            TypedColumn<String>(name: "category", values: ["null", nil, "normal"]),
            TypedColumn<String>(name: "channel", values: ["web", "web", "mobile"]),
            TypedColumn<Double>(name: "revenue", values: [100.0, 200.0, 300.0])
        ])

        let pivoted = try df.pivot(index: "category", columns: "channel", values: "revenue")
        // "null" string and nil should form distinct rows (3 unique index values total)
        #expect(pivoted.shape.rows == 3)
        #expect(pivoted.columns[0].dtype == .utf8)
    }

    @Test("Pivot rejects column value colliding with index column name")
    func testPivotCollidingColumnNameThrows() throws {
        let df = try DataFrame(columns: [
            TypedColumn<String>(name: "date", values: ["2026-01-01", "2026-01-02"]),
            TypedColumn<String>(name: "group", values: ["date", "other"]),
            TypedColumn<Double>(name: "score", values: [10.0, 20.0])
        ])

        #expect(throws: SwiftMLError.self) {
            _ = try df.pivot(index: "date", columns: "group", values: "score")
        }
    }

    @Test("Melt unpivoting heterogeneous types resolves to safe common type without data loss")
    func testMeltHeterogeneousColumnTypes() throws {
        let df = try DataFrame(columns: [
            TypedColumn<Int64>(name: "id", values: [1, 2]),
            TypedColumn<Int64>(name: "age", values: [25, 30]),
            TypedColumn<String>(name: "city", values: ["Kyiv", "Lviv"])
        ])

        let melted = try df.melt(idVars: ["id"], valueVars: ["age", "city"], varName: "attribute", valueName: "val")
        #expect(melted.shape.rows == 4)

        guard let valCol = melted[column: "val"] as? TypedColumn<String> else {
            Issue.record("Expected value column to be resolved as .utf8 string")
            return
        }
        #expect(valCol.values == ["25", "30", "Kyiv", "Lviv"])
    }

    @Test("Melt rejects parameter name collisions with idVars or each other")
    func testMeltNameCollisionsThrow() throws {
        let df = try DataFrame(columns: [
            TypedColumn<Int64>(name: "id", values: [1, 2]),
            TypedColumn<Double>(name: "score", values: [10.0, 20.0])
        ])

        #expect(throws: SwiftMLError.self) {
            _ = try df.melt(idVars: ["id"], valueVars: ["score"], varName: "id", valueName: "val")
        }

        #expect(throws: SwiftMLError.self) {
            _ = try df.melt(idVars: ["id"], valueVars: ["score"], varName: "col", valueName: "col")
        }
    }

    @Test("ChunkedDataFrame streaming join rejects unsupported join kinds")
    func testStreamingJoinUnsupportedKindThrows() async throws {
        let df1 = try DataFrame(columns: [
            TypedColumn<Int64>(name: "id", values: [1, 2])
        ])
        let df2 = try DataFrame(columns: [
            TypedColumn<Int64>(name: "id", values: [2, 3])
        ])

        let chunked = df1.chunked()
        let joinedStream = chunked.join(df2, on: "id", how: .right)

        var didThrow = false
        do {
            for try await _ in joinedStream {}
        } catch {
            didThrow = true
        }
        #expect(didThrow)
    }

    @Test("Pivot preserves original integer index type")
    func testPivotPreservesNumericIndexType() throws {
        let df = try DataFrame(columns: [
            TypedColumn<Int64>(name: "customerID", values: [101, 102]),
            TypedColumn<String>(name: "category", values: ["A", "A"]),
            TypedColumn<Double>(name: "amount", values: [10.0, 20.0])
        ])

        let pivoted = try df.pivot(index: "customerID", columns: "category", values: "amount")
        #expect(pivoted.shape.rows == 2)
        guard let idCol = pivoted[column: "customerID"] as? TypedColumn<Int64> else {
            Issue.record("Expected customerID to retain Int64 type")
            return
        }
        #expect(idCol.values == [101, 102])
    }

    @Test("Melt unpivoting Int64 and Double coerces to Double without precision loss")
    func testMeltHeterogeneousNumericCoercion() throws {
        let df = try DataFrame(columns: [
            TypedColumn<Int64>(name: "id", values: [1, 2]),
            TypedColumn<Int64>(name: "age", values: [25, 30]),
            TypedColumn<Double>(name: "score", values: [9.5, 8.7])
        ])

        let melted = try df.melt(idVars: ["id"], valueVars: ["age", "score"], varName: "metric", valueName: "val")
        #expect(melted.shape.rows == 4)

        guard let valCol = melted[column: "val"] as? TypedColumn<Double> else {
            Issue.record("Expected metric values to coerce to Double")
            return
        }
        #expect(valCol.values == [25.0, 30.0, 9.5, 8.7])
    }

    @Test("Melt rejects duplicate idVars or duplicate valueVars")
    func testMeltDuplicateIdOrValueVarsThrows() throws {
        let df = try DataFrame(columns: [
            TypedColumn<Int64>(name: "id", values: [1, 2]),
            TypedColumn<Double>(name: "val", values: [10.0, 20.0])
        ])

        #expect(throws: SwiftMLError.self) {
            _ = try df.melt(idVars: ["id", "id"], valueVars: ["val"])
        }

        #expect(throws: SwiftMLError.self) {
            _ = try df.melt(idVars: ["id"], valueVars: ["val", "val"])
        }
    }

    @Test("Melt rejects columns present in both idVars and valueVars")
    func testMeltOverlapIdAndValueVarsThrows() throws {
        let df = try DataFrame(columns: [
            TypedColumn<Int64>(name: "id", values: [1, 2]),
            TypedColumn<Double>(name: "val", values: [10.0, 20.0])
        ])

        #expect(throws: SwiftMLError.self) {
            _ = try df.melt(idVars: ["id"], valueVars: ["id", "val"])
        }
    }
}
