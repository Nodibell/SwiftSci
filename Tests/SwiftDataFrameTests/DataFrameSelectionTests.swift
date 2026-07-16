import Testing
import SwiftDataFrame

@Suite("DataFrame Selection")
struct DataFrameSelectionTests {

    private func makeDF() throws -> DataFrame {
        let a = TypedColumn<Int64>(name: "a", values: [1, 2, 3])
        let b = TypedColumn<Double>(name: "b", values: [1.1, 2.2, 3.3])
        let c = TypedColumn<String>(name: "c", values: ["x", "y", "z"])
        return try DataFrame(columns: [a, b, c])
    }

    @Test("select returns only requested columns")
    func selectColumns() throws {
        let df  = try makeDF()
        let sub = try df.select("a", "c")
        #expect(sub.columnNames == ["a", "c"])
        #expect(sub.shape.rows  == 3)
    }

    @Test("select with unknown column throws columnNotFound")
    func selectUnknown() throws {
        let df = try makeDF()
        #expect(throws: DataFrameError.self) {
            try df.select("nonexistent")
        }
    }

    @Test("drop removes specified columns")
    func dropColumns() throws {
        let df  = try makeDF()
        let sub = try df.drop("b")
        #expect(sub.columnNames == ["a", "c"])
    }

    @Test("head returns first N rows")
    func headRows() throws {
        let df = try makeDF()
        let h  = df.head(2)
        #expect(h.shape.rows == 2)
        let col = h[column: "a", as: Int64.self]
        #expect(col?.values == [1, 2])
    }

    @Test("head(0) returns empty DataFrame")
    func headZero() throws {
        let df = try makeDF()
        #expect(df.head(0).shape.rows == 0)
    }

    @Test("tail returns last N rows")
    func tailRows() throws {
        let df = try makeDF()
        let t  = df.tail(2)
        #expect(t.shape.rows == 2)
        let col = t[column: "a", as: Int64.self]
        #expect(col?.values == [2, 3])
    }

    @Test("sample with seed is reproducible")
    func sampleReproducible() throws {
        let df = try makeDF()
        let s1 = df.sample(n: 2, seed: 42)
        let s2 = df.sample(n: 2, seed: 42)
        #expect(s1.shape.rows == s2.shape.rows)
        // Same seed must produce same rows
        let v1 = s1[column: "a", as: Int64.self]?.values
        let v2 = s2[column: "a", as: Int64.self]?.values
        #expect(v1 == v2)
    }
}
