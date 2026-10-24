import Testing
@testable import SwiftDataFrame

@Suite("Compact numeric consumer ownership")
struct CompactNumericConsumerTests {
    @Test("Preparing a batch shares dense values and validity while preserving snapshots")
    func batchSharesStorage() throws {
        let column = denseColumn([1, nil, -0.0, .nan, 5])
        guard case .dense(let values, let validity) = column.storage else {
            Issue.record("Fixture must use dense storage")
            return
        }
        let frame = try DataFrame(columns: [column])
        var batch = try frame.prepareNumericBatch(["x"])
        #expect(sameStorage(values, batch.columns[0].values))
        let sourceBits = try #require(validity)
        let batchBits = try #require(batch.columns[0].validity)
        #expect(sameStorage(sourceBits, batchBits))
        try batch.updateColumn(at: 0, rows: [0, 1], values: [99, 100])
        #expect(column[0] == 1)
        #expect(column[1] == nil)
        #expect(batch[0, 0] == 99)
        #expect(batch[1, 0] == 100)
    }

    @Test("Single-column exports reuse present dense values in either order")
    func singleColumnSharesStorage() throws {
        let expected: [Double?] = [1, -0.0, Double(bitPattern: 0x7ff8000000000042), .infinity]
        let column = denseColumn(expected)
        guard case .dense(let values, _) = column.storage else {
            Issue.record("Fixture must use dense storage")
            return
        }
        let frame = try DataFrame(columns: [column])
        for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
            for missing in [NumericMissingValuePolicy.nan, .reject] {
                let matrix = try frame.prepareNumericMatrix(["x"], order: order, missing: missing)
                #expect(sameStorage(values, matrix.values))
                compare(matrix.values, expected.map { $0! })
            }
        }
        #expect(sameStorage(values, try frame.toFlatFeatureMatrix(["x"]).flat))
        #expect(sameStorage(values, try frame.toTargetVector("x")))
    }

    @Test("Native packing agrees with optional storage for mixed and repeated columns")
    func mixedMatrices() throws {
        let expected: [Double?] = [1, nil, -0.0, Double(bitPattern: 0x7ff8000000000042), .infinity, -.infinity]
        let integer = TypedColumn<Int64>(name: "i", values: [9_007_199_254_740_993, 2, 3, 4, 5, 6])
        let boolean = TypedColumn<Bool>(name: "b", values: [true, false, nil, true, false, nil])
        let native = try DataFrame(columns: [denseColumn(expected), integer, boolean])
        let optional = try DataFrame(columns: [TypedColumn(name: "x", values: expected), integer, boolean])
        let names = ["x", "b", "i", "x"]
        let selections: [[Int]?] = [nil, [], [5, 1, 1, 0], [0, 2, 3]]
        for rows in selections {
            for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
                let actual = try native.prepareNumericMatrix(names, rows: rows, order: order)
                let reference = try optional.prepareNumericMatrix(names, rows: rows, order: order)
                compare(actual.values, reference.values)
                #expect(actual.originalRowIndices == reference.originalRowIndices)
            }
        }
        compare(try native.toFlatFeatureMatrix(names).flat, try optional.toFlatFeatureMatrix(names).flat)
        let nativeNested = try native.toFeatureMatrix(names)
        let optionalNested = try optional.toFeatureMatrix(names)
        for (actual, reference) in zip(nativeNested, optionalNested) { compare(actual, reference) }
        let exported = try native.prepareNumericBatch(["x"]).columnValues(at: 0)
        #expect(exported[1] == nil)
        #expect(exported[3]?.bitPattern == expected[3]?.bitPattern)
    }

    @Test("Rejecting missing values checks only selected rows and still permits NaN")
    func rejectionSelection() throws {
        let frame = try DataFrame(columns: [denseColumn([1, nil, -0.0, .nan])])
        let selected = try frame.prepareNumericMatrix(["x"], rows: [3, 0], missing: .reject)
        #expect(selected.values[0].isNaN)
        #expect(selected.values[1] == 1)
        let empty = try frame.prepareNumericMatrix(["x"], rows: [], missing: .reject)
        #expect(empty.values.isEmpty)
        #expect(throws: SwiftMLError.self) { try frame.prepareNumericMatrix(["x"], missing: .reject) }
        #expect(throws: SwiftMLError.self) { try frame.prepareNumericMatrix(["x"], rows: [1], missing: .reject) }
    }

    private func denseColumn(_ values: [Double?]) -> TypedColumn<Double> {
        var builder = CSVCompactDoubleBuilder(capacity: values.count)
        for value in values { builder.append(value) }
        return builder.column(named: "x")
    }

    private func sameStorage<T>(_ lhs: [T], _ rhs: [T]) -> Bool {
        lhs.withUnsafeBufferPointer { left in rhs.withUnsafeBufferPointer { right in
            left.baseAddress == right.baseAddress
        } }
    }

    private func compare(_ actual: [Double], _ expected: [Double]) {
        #expect(actual.count == expected.count)
        for (actual, expected) in zip(actual, expected) { #expect(actual.bitPattern == expected.bitPattern) }
    }
}
