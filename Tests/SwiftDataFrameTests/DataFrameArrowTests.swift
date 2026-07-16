import Testing
import Foundation
import Arrow
@testable import SwiftDataFrame

@Suite("DataFrame Arrow Integration Tests")
struct DataFrameArrowTests {
    
    @Test("Convert ArrowTable to DataFrame")
    func testArrowToDataFrame() throws {
        // Build raw Arrow arrays
        let intBuilder = try ArrowArrayBuilders.loadBuilder(Int32.self)
        intBuilder.appendAny(Int32(10))
        intBuilder.appendAny(nil)
        intBuilder.appendAny(Int32(30))
        let intHolder = try intBuilder.toHolder()
        
        let strBuilder = try ArrowArrayBuilders.loadBuilder(String.self)
        strBuilder.appendAny("A")
        strBuilder.appendAny("B")
        strBuilder.appendAny(nil)
        let strHolder = try strBuilder.toHolder()
        
        let doubleBuilder = try ArrowArrayBuilders.loadBuilder(Double.self)
        doubleBuilder.appendAny(1.1)
        doubleBuilder.appendAny(2.2)
        doubleBuilder.appendAny(3.3)
        let doubleHolder = try doubleBuilder.toHolder()
        
        let rbBuilder = RecordBatch.Builder()
        rbBuilder.addColumn("col_int", arrowArray: intHolder)
        rbBuilder.addColumn("col_str", arrowArray: strHolder)
        rbBuilder.addColumn("col_double", arrowArray: doubleHolder)
        
        let rb = try rbBuilder.finish().get()
        let table = try ArrowTable.from(recordBatches: [rb]).get()
        
        // Convert
        let df = try DataFrame(arrowTable: table)
        #expect(df.shape.rows == 3)
        #expect(df.shape.columns == 3)
        
        // Assert values
        let ints = df[column: "col_int", as: Int32.self]?.values
        #expect(ints == [10, nil, 30])
        
        let strs = df[column: "col_str", as: String.self]?.values
        #expect(strs == ["A", "B", nil])
        
        let doubles = df[column: "col_double", as: Double.self]?.values
        #expect(doubles == [1.1, 2.2, 3.3])
    }
    
    @Test("Convert DataFrame to ArrowTable")
    func testDataFrameToArrow() throws {
        let name = TypedColumn<String>(name: "name", values: ["Alice", "Bob", nil])
        let age  = TypedColumn<Int64>(name: "age", values: [25, nil, 35])
        let df = try DataFrame(columns: [name, age])
        
        // Convert
        let table = try df.toArrowTable()
        #expect(table.rowCount == 3)
        #expect(table.columnCount == 2)
        #expect(table.columns[0].name == "name")
        #expect(table.columns[1].name == "age")
        
        // Verify types
        #expect(table.columns[0].type.id == .string)
        #expect(table.columns[1].type.id == .int64)
        
        // Verify values from Arrow table column data
        let nameChunked: ChunkedArray<String> = table.columns[0].data()
        #expect(nameChunked[0] == "Alice")
        #expect(nameChunked[1] == "Bob")
        #expect(nameChunked[2] == nil)
        
        let ageChunked: ChunkedArray<Int64> = table.columns[1].data()
        #expect(ageChunked[0] == 25)
        #expect(ageChunked[1] == nil)
        #expect(ageChunked[2] == 35)
    }

    // MARK: – Zero-Copy Invariant Tests

    /// Verifies that large Arrow → DataFrame conversions preserve value count exactly
    /// (sanity check that the bridging loop reads all elements correctly)
    /// and that DataFrame → ArrowTable round-trip produces a structurally equivalent table.
    @Test("Arrow round-trip preserves row/column counts for large buffers")
    func testArrowRoundTripLargeBuffer() throws {
        let rowCount = 10_000

        // Build large Arrow arrays
        let doubleBuilder = try ArrowArrayBuilders.loadBuilder(Double.self)
        let int64Builder  = try ArrowArrayBuilders.loadBuilder(Int64.self)
        let strBuilder    = try ArrowArrayBuilders.loadBuilder(String.self)

        for i in 0..<rowCount {
            doubleBuilder.appendAny(Double(i) * 0.001)
            int64Builder.appendAny(Int64(i))
            strBuilder.appendAny(i % 100 == 0 ? nil : "row_\(i)")
        }

        let dblHolder = try doubleBuilder.toHolder()
        let intHolder = try int64Builder.toHolder()
        let strHolder = try strBuilder.toHolder()

        let rbBuilder = RecordBatch.Builder()
        rbBuilder.addColumn("value",    arrowArray: dblHolder)
        rbBuilder.addColumn("index",    arrowArray: intHolder)
        rbBuilder.addColumn("label",    arrowArray: strHolder)

        let rb    = try rbBuilder.finish().get()
        let table = try ArrowTable.from(recordBatches: [rb]).get()

        // ── Arrow → DataFrame ──────────────────────────────────────────────
        let df = try DataFrame(arrowTable: table)
        #expect(df.shape.rows    == rowCount)
        #expect(df.shape.columns == 3)

        // Spot-check a few values
        let vals  = df[column: "value",  as: Double.self]?.values
        let idxs  = df[column: "index",  as: Int64.self]?.values
        let labels = df[column: "label", as: String.self]?.values

        #expect(vals?[0]    == 0.0)
        #expect(vals?[999]  == 0.999)
        #expect(idxs?[0]    == 0)
        #expect(idxs?[9999] == 9_999)
        #expect(labels?[0]  == nil)          // i % 100 == 0 → nil
        #expect(labels?[1]  == "row_1")

        // ── DataFrame → ArrowTable (round-trip) ────────────────────────────
        let roundTripped = try df.toArrowTable()
        #expect(roundTripped.rowCount    == rowCount)
        #expect(roundTripped.columnCount == 3)

        let rtVals: ChunkedArray<Double> = roundTripped.columns[0].data()
        #expect(rtVals[0] == 0.0)
    }

    /// Verifies that an Arrow table with many null values is correctly reflected
    /// as nil entries in the DataFrame (null bitmap tracking).
    @Test("Arrow null bitmap transfers correctly to DataFrame null count")
    func testArrowNullBitmapPreservation() throws {
        let rowCount = 200
        let builder = try ArrowArrayBuilders.loadBuilder(Double.self)
        var expectedNullCount = 0

        for i in 0..<rowCount {
            if i % 5 == 0 {
                builder.appendAny(nil)          // every 5th row is null
                expectedNullCount += 1
            } else {
                builder.appendAny(Double(i))
            }
        }

        let holder = try builder.toHolder()
        let rbBuilder = RecordBatch.Builder()
        rbBuilder.addColumn("data", arrowArray: holder)
        let rb    = try rbBuilder.finish().get()
        let table = try ArrowTable.from(recordBatches: [rb]).get()

        let df = try DataFrame(arrowTable: table)
        let col = df[column: "data", as: Double.self]
        #expect(col != nil)
        #expect(col!.nullCount == expectedNullCount)
        #expect(col!.count     == rowCount)

        // Non-null values should be exactly as inserted
        #expect(col!.values[1]  == 1.0)
        #expect(col!.values[5]  == nil)
        #expect(col!.values[10] == nil)
        #expect(col!.values[11] == 11.0)
    }
}
