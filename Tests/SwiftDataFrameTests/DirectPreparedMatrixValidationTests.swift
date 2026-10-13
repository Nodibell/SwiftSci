import Testing
import SwiftDataFrame

@Suite("Direct matrix validation")
struct DirectPreparedMatrixValidationTests {
    @Test func rejectionUsesSelectedNilNotValidNaN() throws {
        let doubles: [Double?] = [.nan, nil, -0.0]
        let integers: [Int64?] = [1, nil, 3]
        let booleans: [Bool?] = [true, nil, false]
        let frame = try DataFrame(columns: [
            TypedColumn<Double>(name: "d", values: doubles),
            TypedColumn<Int64>(name: "i", values: integers),
            TypedColumn<Bool>(name: "b", values: booleans)
        ])
        for name in ["d", "i", "b"] {
            for order in [NumericMatrixOrder.rowMajor, .columnMajor] {
                let error = SwiftMLError.invalidParameter("Column \(name) contains missing values")
                #expect(throws: error) { try frame.prepareNumericMatrix([name], order: order, missing: .reject) }
                #expect(throws: error) {
                    try frame.prepareNumericMatrix([name], rows: [2, 1], order: order, missing: .reject)
                }
                let valid = try frame.prepareNumericMatrix([name], rows: [0, 2, 0], order: order, missing: .reject)
                #expect(valid.rowCount == 3)
                if name == "d" { #expect(valid.values[0].isNaN && valid.values[1].sign == .minus) }
                #expect(try frame.prepareNumericMatrix([name], rows: [], missing: .reject).values.isEmpty)
            }
        }
    }

    @Test func invalidRowsAndNamesThrow() throws {
        let frame = try DataFrame(columns: [
            TypedColumn<Double>(name: "x", values: [1, 2]),
            TypedColumn<String>(name: "text", values: ["a", "b"])
        ])
        let invalid = SwiftMLError.invalidParameter("Row selection contains an out-of-bounds index")
        for rows in [[-1], [2], [0, Int.max]] {
            #expect(throws: invalid) { try frame.prepareNumericMatrix(["x"], rows: rows) }
            #expect(throws: invalid) { try frame.prepareNumericMatrix([], rows: rows) }
        }
        #expect(throws: SwiftMLError.columnNotFound("missing")) {
            try frame.prepareNumericMatrix(["text", "missing"], rows: [])
        }
        #expect(throws: SwiftMLError.castFailed(column: "text", targetType: "Double")) {
            try frame.prepareNumericMatrix(["text"], rows: [])
        }
    }
}
