import Testing
@testable import SwiftDataFrame

@Suite("Prepared numeric batches")
struct PreparedNumericBatchTests {
    @Test func validityAcrossWords() throws {
        var values = (0..<130).map { Optional(Double($0)) }
        let missing = [0, 1, 63, 64, 65, 127, 128]
        for row in missing { values[row] = nil }
        values[2] = .nan
        let frame = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: values)])
        let batch = try frame.prepareNumericBatch(["x"])
        #expect(batch.nullCount(inColumn: 0) == missing.count)
        #expect(batch[2, 0]?.isNaN == true)
        for row in values.indices {
            #expect((batch[row, 0] == nil) == missing.contains(row))
            if let expected = values[row], !expected.isNaN { #expect(batch[row, 0] == expected) }
        }
        #expect(batch.payloadByteCount == 130 * 8 + 3 * 8)
    }

    @Test func denseAndEmptyStorage() throws {
        let frame = try DataFrame(columns: [TypedColumn<Int64>(name: "i", values: [1, 2]), TypedColumn<Bool>(name: "b", values: [true, false])])
        let batch = try frame.prepareNumericBatch(["b", "i", "b"])
        #expect(batch.columnNames == ["b", "i", "b"])
        #expect(batch.columnValues(at: 0) == [1, 0])
        #expect(batch.columnValues(at: 1) == [1, 2])
        #expect(batch.payloadByteCount == 2 * 3 * 8)
        let noColumns = try frame.prepareNumericBatch([])
        #expect(noColumns.rowCount == 2 && noColumns.columnCount == 0)
        let empty = try DataFrame(columns: [TypedColumn<Double>(name: "x", values: [Double]())]).prepareNumericBatch(["x"])
        #expect(empty.rowCount == 0 && empty.columnValues(at: 0).isEmpty)
    }

    @Test func preparationUsesExistingConversionRules() throws {
        let frame = try DataFrame(columns: [TypedColumn<Float>(name: "x", values: [1])])
        #expect(throws: SwiftMLError.columnNotFound("missing")) { try frame.prepareNumericBatch(["x", "missing"]) }
        #expect(throws: SwiftMLError.castFailed(column: "x", targetType: "Double")) { try frame.prepareNumericBatch(["x"]) }
    }
}
