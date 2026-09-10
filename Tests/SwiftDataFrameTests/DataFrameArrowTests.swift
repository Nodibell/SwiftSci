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

    @Test("ArrowNullStrategy converts nulls to NaN or Zero")
    func testArrowNullStrategy() throws {
        let builder = try ArrowArrayBuilders.loadBuilder(Double.self)
        builder.appendAny(10.0)
        builder.appendAny(nil)
        builder.appendAny(30.0)
        let holder = try builder.toHolder()

        let intBuilder = try ArrowArrayBuilders.loadBuilder(Int32.self)
        intBuilder.appendAny(Int32(1))
        intBuilder.appendAny(nil)
        intBuilder.appendAny(Int32(3))
        let intHolder = try intBuilder.toHolder()

        let rbBuilder = RecordBatch.Builder()
        rbBuilder.addColumn("d", arrowArray: holder)
        rbBuilder.addColumn("i", arrowArray: intHolder)
        let rb = try rbBuilder.finish().get()
        let table = try ArrowTable.from(recordBatches: [rb]).get()

        // 1. Default / .preserve
        let dfPreserve = try DataFrame(arrowTable: table, nullStrategy: .preserve)
        let dPreserve = dfPreserve[column: "d", as: Double.self]?.values
        let iPreserve = dfPreserve[column: "i", as: Int32.self]?.values
        #expect(dPreserve?[1] == nil)
        #expect(iPreserve?[1] == nil)

        // 2. .nan strategy
        let dfNaN = try DataFrame(arrowTable: table, nullStrategy: .nan)
        let dNaN = dfNaN[column: "d", as: Double.self]?.values
        #expect(dNaN?[0] == 10.0)
        #expect(dNaN?[1]?.isNaN == true)
        #expect(dNaN?[2] == 30.0)

        // 3. .zero strategy
        let dfZero = try DataFrame(arrowTable: table, nullStrategy: .zero)
        let dZero = dfZero[column: "d", as: Double.self]?.values
        let iZero = dfZero[column: "i", as: Int32.self]?.values
        #expect(dZero == [10.0, 0.0, 30.0])
        #expect(iZero == [1, 0, 3])
    }

    @Test("ArrowTableBridge nullStrategy for Int64, Float, and Bool columns")
    func testArrowNullStrategyExtendedTypes() throws {
        let i64Builder = try ArrowArrayBuilders.loadBuilder(Int64.self)
        i64Builder.appendAny(Int64(100))
        i64Builder.appendAny(nil)
        let i64Holder = try i64Builder.toHolder()

        let fBuilder = try ArrowArrayBuilders.loadBuilder(Float.self)
        fBuilder.appendAny(Float(1.5))
        fBuilder.appendAny(nil)
        let fHolder = try fBuilder.toHolder()

        let bBuilder = try ArrowArrayBuilders.loadBuilder(Bool.self)
        bBuilder.appendAny(true)
        bBuilder.appendAny(nil)
        let bHolder = try bBuilder.toHolder()

        let rbBuilder = RecordBatch.Builder()
        rbBuilder.addColumn("i64", arrowArray: i64Holder)
        rbBuilder.addColumn("f", arrowArray: fHolder)
        rbBuilder.addColumn("b", arrowArray: bHolder)
        let rb = try rbBuilder.finish().get()
        let table = try ArrowTable.from(recordBatches: [rb]).get()

        // .nan strategy
        let dfNaN = try DataFrame(arrowTable: table, nullStrategy: .nan)
        let fNaN = dfNaN[column: "f", as: Float.self]?.values
        #expect(fNaN?[0] == 1.5)
        #expect(fNaN?[1]?.isNaN == true)

        // .zero strategy
        let dfZero = try DataFrame(arrowTable: table, nullStrategy: .zero)
        let i64Zero = dfZero[column: "i64", as: Int64.self]?.values
        let fZero = dfZero[column: "f", as: Float.self]?.values
        let bZero = dfZero[column: "b", as: Bool.self]?.values
        #expect(i64Zero == [100, 0])
        #expect(fZero == [1.5, 0.0])
        #expect(bZero == [true, false])
    }

    @Test("ArrowDataBuffer owner retention and slicing zero-copy invariant")
    func testArrowDataBufferOwnerAndSlice() throws {
        final class MockOwner: @unchecked Sendable {}
        let owner = MockOwner()
        let memory = [10.0, 20.0, 30.0, 40.0]

        memory.withUnsafeBytes { rawBuffer in
            guard let baseAddr = rawBuffer.baseAddress else { return }
            let buffer = ArrowDataBuffer<Double>(
                rawPointer: baseAddr,
                byteCount: rawBuffer.count,
                elementCount: memory.count,
                owner: owner
            )
            #expect(buffer.elementCount == 4)
            #expect(buffer.byteCount == 4 * MemoryLayout<Double>.stride)

            buffer.withUnsafeBytes { sliceBuf in
                let ptr = sliceBuf.bindMemory(to: Double.self)
                #expect(ptr[0] == 10.0)
                #expect(ptr[3] == 40.0)
            }

            let sliced = buffer.slice(from: 1, count: 2)
            #expect(sliced.elementCount == 2)
            sliced.withUnsafeBytes { sliceBuf in
                let ptr = sliceBuf.bindMemory(to: Double.self)
                #expect(ptr[0] == 20.0)
                #expect(ptr[1] == 30.0)
            }
        }
    }
}
