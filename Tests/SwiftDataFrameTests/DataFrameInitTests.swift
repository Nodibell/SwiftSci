import Testing
import SwiftDataFrame

@Suite("DataFrame Initialisation")
struct DataFrameInitTests {

    @Test("Empty columns list throws emptySchema")
    func emptyColumnsThrows() throws {
        #expect(throws: DataFrameError.emptySchema) {
            try DataFrame(columns: [])
        }
    }

    @Test("Mismatched column lengths throw columnLengthMismatch")
    func columnLengthMismatch() throws {
        let col1 = TypedColumn<Int64>(name: "a", values: [1, 2, 3])
        let col2 = TypedColumn<Int64>(name: "b", values: [1, 2])
        #expect(throws: DataFrameError.self) {
            try DataFrame(columns: [col1, col2])
        }
    }

    @Test("Duplicate column name throws")
    func duplicateColumnName() throws {
        let col1 = TypedColumn<Int64>(name: "a", values: [1, 2])
        let col2 = TypedColumn<Int64>(name: "a", values: [3, 4])
        #expect(throws: DataFrameError.self) {
            try DataFrame(columns: [col1, col2])
        }
    }

    @Test("Shape is correct")
    func shapeIsCorrect() throws {
        let col1 = TypedColumn<Int64>(name: "x", values: [1, 2, 3])
        let col2 = TypedColumn<Double>(name: "y", values: [1.0, 2.0, 3.0])
        let df   = try DataFrame(columns: [col1, col2])
        #expect(df.shape.rows    == 3)
        #expect(df.shape.columns == 2)
    }

    @Test("Column names order is preserved")
    func columnNamesOrder() throws {
        let cols = ["z", "a", "m"].map { TypedColumn<Int64>(name: $0, values: [0]) }
        let df   = try DataFrame(columns: cols)
        #expect(df.columnNames == ["z", "a", "m"])
    }

    @Test("All supported dtypes initialise correctly")
    func allDTypes() throws {
        let cols: [any AnyColumn] = [
            TypedColumn<Int32>(name: "i32",  values: [1]),
            TypedColumn<Int64>(name: "i64",  values: [1]),
            TypedColumn<Float>(name: "f32",  values: [1.0]),
            TypedColumn<Double>(name: "f64", values: [1.0]),
            TypedColumn<Bool>(name: "bool",  values: [true]),
            TypedColumn<String>(name: "str", values: ["hi"]),
        ]
        let df = try DataFrame(columns: cols)
        #expect(df.shape.columns == 6)
    }

    @Test("Nullable column tracks nullCount")
    func nullCount() throws {
        let col = TypedColumn<Int64>(name: "n", values: [1, nil, 3, nil])
        #expect(col.nullCount == 2)
    }
}
