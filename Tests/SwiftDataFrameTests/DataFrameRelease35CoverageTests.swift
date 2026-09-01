import Testing
import Foundation
@testable import SwiftDataFrame

@Suite("SwiftDataFrame Release 3.5.0 Coverage Tests")
struct DataFrameRelease35CoverageTests {

    // MARK: - ParquetWriter Typed Fast-Paths & Nulls

    @Test("ParquetWriter encodes all typed columns without nulls")
    func testParquetWriterAllTypedColumnsNoNulls() async throws {
        let cInt64 = TypedColumn<Int64>(name: "c_i64", values: [100, 200, 300])
        let cInt32 = TypedColumn<Int32>(name: "c_i32", values: [10, 20, 30])
        let cDouble = TypedColumn<Double>(name: "c_dbl", values: [1.5, 2.5, 3.5])
        let cFloat = TypedColumn<Float>(name: "c_flt", values: [10.5, 20.5, 30.5])
        let cBool = TypedColumn<Bool>(name: "c_bool", values: [true, false, true])
        let cStr = TypedColumn<String>(name: "c_str", values: ["alpha", "beta", "gamma"])
        let cDate = TypedColumn<Date>(name: "c_date", values: [
            Date(timeIntervalSince1970: 86400 * 10),
            Date(timeIntervalSince1970: 86400 * 20),
            Date(timeIntervalSince1970: 86400 * 30)
        ])

        let df = try DataFrame(columns: [cInt64, cInt32, cDouble, cFloat, cBool, cStr, cDate])
        let tempURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("test_\(UUID().uuidString).parquet")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try await ParquetWriter.write(dataFrame: df, to: tempURL)
        let data = try Data(contentsOf: tempURL)

        #expect(!data.isEmpty)
        // Check Parquet magic header "PAR1"
        let header = Array(data.prefix(4))
        #expect(String(bytes: header, encoding: .utf8) == "PAR1")
        let footer = Array(data.suffix(4))
        #expect(String(bytes: footer, encoding: .utf8) == "PAR1")
    }

    @Test("ParquetWriter encodes nullable columns with missing values")
    func testParquetWriterNullableColumns() async throws {
        let cInt64 = TypedColumn<Int64>(name: "c_i64", values: [Optional(100), nil, Optional(300)])
        let cInt32 = TypedColumn<Int32>(name: "c_i32", values: [nil, Optional(20), Optional(30)])
        let cDouble = TypedColumn<Double>(name: "c_dbl", values: [Optional(1.5), Optional(2.5), nil])
        let cFloat = TypedColumn<Float>(name: "c_flt", values: [nil, nil, Optional(30.5)])
        let cBool = TypedColumn<Bool>(name: "c_bool", values: [Optional(true), nil, Optional(false)])
        let cStr = TypedColumn<String>(name: "c_str", values: [Optional("a"), nil, Optional("c")])
        let cDate = TypedColumn<Date>(name: "c_date", values: [
            Optional(Date(timeIntervalSince1970: 86400)),
            nil,
            Optional(Date(timeIntervalSince1970: 86400 * 3))
        ])

        let df = try DataFrame(columns: [cInt64, cInt32, cDouble, cFloat, cBool, cStr, cDate])
        let tempURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("test_nulls_\(UUID().uuidString).parquet")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try await ParquetWriter.write(dataFrame: df, to: tempURL)
        let data = try Data(contentsOf: tempURL)

        #expect(!data.isEmpty)
        let footer = Array(data.suffix(4))
        #expect(String(bytes: footer, encoding: .utf8) == "PAR1")
    }

    // MARK: - DataFrame + Flat Feature Matrix

    @Test("DataFrame toFlatFeatureMatrix with Double, Int64, and Bool")
    func testToFlatFeatureMatrix() throws {
        let colDbl = TypedColumn<Double>(name: "f1", values: [1.0, 2.0, 3.0])
        let colInt = TypedColumn<Int64>(name: "f2", values: [10, 20, 30])
        let colBool = TypedColumn<Bool>(name: "f3", values: [true, false, true])
        let df = try DataFrame(columns: [colDbl, colInt, colBool])

        let (flat, rows, cols) = try df.toFlatFeatureMatrix(["f1", "f2", "f3"])
        #expect(rows == 3)
        #expect(cols == 3)
        #expect(flat.count == 9)

        // Row 0: 1.0, 10.0, 1.0
        #expect(flat[0] == 1.0)
        #expect(flat[1] == 10.0)
        #expect(flat[2] == 1.0)

        // Row 1: 2.0, 20.0, 0.0
        #expect(flat[3] == 2.0)
        #expect(flat[4] == 20.0)
        #expect(flat[5] == 0.0)
    }

    @Test("DataFrame toFlatFeatureMatrix handles nulls as NaN")
    func testToFlatFeatureMatrixNulls() throws {
        let colDbl = TypedColumn<Double>(name: "f1", values: [Optional(1.0), nil])
        let colInt = TypedColumn<Int64>(name: "f2", values: [nil, Optional(20)])
        let colBool = TypedColumn<Bool>(name: "f3", values: [Optional(true), nil])
        let df = try DataFrame(columns: [colDbl, colInt, colBool])

        let (flat, rows, cols) = try df.toFlatFeatureMatrix(["f1", "f2", "f3"])
        #expect(rows == 2)
        #expect(cols == 3)
        #expect(flat[1].isNaN)
        #expect(flat[3].isNaN)
        #expect(flat[5].isNaN)
    }

    @Test("DataFrame toFlatFeatureMatrix error handling")
    func testToFlatFeatureMatrixErrors() throws {
        let colStr = TypedColumn<String>(name: "txt", values: ["a", "b"])
        let colDbl = TypedColumn<Double>(name: "val", values: [1.0, 2.0])
        let df = try DataFrame(columns: [colStr, colDbl])

        // Column not found
        #expect(throws: SwiftMLError.self) {
            try df.toFlatFeatureMatrix(["val", "missing_col"])
        }

        // Unsupported cast (String cannot be cast to Double)
        #expect(throws: SwiftMLError.self) {
            try df.toFlatFeatureMatrix(["val", "txt"])
        }
    }

    // MARK: - DataFrame Rows & Sequence & Dictionary Accessor

    @Test("DataFrame rowDictionary(at:) accessor")
    func testRowDictionaryAccessor() throws {
        let c1 = TypedColumn<Int64>(name: "id", values: [1, 2])
        let c2 = TypedColumn<String>(name: "name", values: ["Alice", "Bob"])
        let df = try DataFrame(columns: [c1, c2])

        let row0 = df.rowDictionary(at: 0)
        #expect(row0["id"] as? Int64 == 1)
        #expect(row0["name"] as? String == "Alice")

        let outOfBounds = df.rowDictionary(at: 99)
        #expect(outOfBounds.isEmpty)

        let negative = df.rowDictionary(at: -1)
        #expect(negative.isEmpty)
    }

    @Test("DataFrame lightweight row(at:) and rows sequence iteration")
    func testDataFrameRowAndSequence() throws {
        let c1 = TypedColumn<Double>(name: "x", values: [10.0, 20.0, 30.0])
        let c2 = TypedColumn<String>(name: "label", values: [Optional("L1"), nil, Optional("L3")])
        let df = try DataFrame(columns: [c1, c2])

        // Single row inspection
        let r0 = df.row(at: 0)
        #expect(r0.double("x") == 10.0)
        #expect(r0.string("label") == "L1")
        #expect(r0["x", as: Double.self] == 10.0)
        #expect(r0["x"] as? Double == 10.0)
        #expect(!r0.isNull(column: "x"))

        let r1 = df.row(at: 1)
        #expect(r1.isNull(column: "label"))

        // Sequence iteration
        var collectedX = [Double]()
        var count = 0
        for row in df.rows {
            if let v = row.double("x") {
                collectedX.append(v)
            }
            count += 1
        }
        #expect(count == 3)
        #expect(df.rows.count == 3)
        #expect(collectedX == [10.0, 20.0, 30.0])
    }

    // MARK: - Schema FieldMap Fast Lookup

    @Test("Schema subscript dictionary lookup")
    func testSchemaLookup() {
        let f1 = Schema.Field(name: "id", dtype: .int64, nullable: false)
        let f2 = Schema.Field(name: "score", dtype: .float64, nullable: true)
        let schema = Schema(fields: [f1, f2])

        #expect(schema.columnNames == ["id", "score"])
        #expect(schema["id"]?.dtype == .int64)
        #expect(schema["score"]?.nullable == true)
        #expect(schema["non_existent"] == nil)
    }
}
