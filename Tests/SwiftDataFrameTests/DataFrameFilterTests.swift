import Testing
import SwiftDataFrame

@Suite("DataFrame Filter & Transform")
struct DataFrameFilterTests {

    private func makeDF() throws -> DataFrame {
        let age  = TypedColumn<Int64>(name: "age",  values: [25, 30, 35, 40, nil])
        let name = TypedColumn<String>(name: "name", values: ["Alice", "Bob", "Carol", "Dave", "Eve"])
        return try DataFrame(columns: [age, name])
    }

    @Test("filter(predicate:) keeps matching rows")
    func filterPredicate() throws {
        let df  = try makeDF()
        let res = df.filter { row in
            (row.value(column: "age", as: Int64.self) ?? 0) >= 35
        }
        #expect(res.shape.rows == 2)
    }

    @Test("filter on empty DataFrame returns empty")
    func filterEmpty() {
        let result = DataFrame.empty.filter { _ in true }
        #expect(result.shape.rows == 0)
    }

    @Test("filter(column:where:) greaterThan")
    func filterColumnGreaterThan() throws {
        let df  = try makeDF()
        let res = try df.filter(column: "age", where: .greaterThan(Int64(29)))
        // ages 30, 35, 40 pass; nil excluded; result = 3 rows
        #expect(res.shape.rows == 3)
    }

    @Test("filter isNull keeps only null rows")
    func filterIsNull() throws {
        let df  = try makeDF()
        let res = try df.filter(column: "age", where: .isNull)
        #expect(res.shape.rows == 1)
    }

    @Test("sortBy ascending")
    func sortByAscending() throws {
        let df     = try makeDF()
        let sorted = try df.sortBy("age", ascending: true)
        let ages   = sorted[column: "age", as: Int64.self]?.values.compactMap { $0 }
        #expect(ages == [25, 30, 35, 40])
    }

    @Test("sortBy null values appear last")
    func sortByNullLast() throws {
        let df   = try makeDF()
        let s    = try df.sortBy("age", ascending: true)
        let vals = s[column: "age", as: Int64.self]?.values
        // The last element is a null value — it's .some(nil) in Optional<Optional<Int64>>
        #expect(vals?.last == .some(nil))
    }

    @Test("withColumn adds new column")
    func withColumnAdds() throws {
        let df     = try makeDF()
        let newCol = TypedColumn<Bool>(name: "senior", values: [false, false, true, true, nil])
        let df2    = try df.withColumn("senior", column: newCol)
        #expect(df2.columnNames.contains("senior"))
        #expect(df2.shape.columns == 3)
    }

    @Test("withColumn overwrites existing column")
    func withColumnOverwrites() throws {
        let df     = try makeDF()
        let newCol = TypedColumn<Int64>(name: "age", values: [0, 0, 0, 0, 0])
        let df2    = try df.withColumn("age", column: newCol)
        #expect(df2.shape.columns == 2) // still 2 columns
    }

    @Test("renameColumn succeeds")
    func renameColumn() throws {
        let df  = try makeDF()
        let df2 = try df.renameColumn("age", to: "years")
        #expect(df2.columnNames.contains("years"))
        #expect(!df2.columnNames.contains("age"))
    }

    @Test("castColumn Int64 → Double succeeds")
    func castIntToDouble() throws {
        let df  = try makeDF()
        let df2 = try df.castColumn("age", to: Double.self)
        let col = df2[column: "age", as: Double.self]
        #expect(col != nil)
        #expect(col?.values.compactMap { $0 }.first == 25.0)
    }

    @Test("withLaggedColumn shifts values properly with null filling")
    func withLaggedColumn() throws {
        let df = try makeDF()
        
        // Positive lag by 1
        let dfLag1 = try df.withLaggedColumn(column: "age", by: 1, newName: "age_lag_1")
        let colLag1 = dfLag1[column: "age_lag_1", as: Int64.self]
        #expect(colLag1 != nil)
        #expect(colLag1?.values[0] == nil)
        #expect(colLag1?.values[1] == 25)
        #expect(colLag1?.values[2] == 30)
        #expect(colLag1?.values[3] == 35)
        #expect(colLag1?.values[4] == 40)
        
        // Negative lag by 2 (leads values back by 2)
        let dfLead2 = try df.withLaggedColumn(column: "age", by: -2, newName: "age_lead_2")
        let colLead2 = dfLead2[column: "age_lead_2", as: Int64.self]
        #expect(colLead2 != nil)
        #expect(colLead2?.values[0] == 35)
        #expect(colLead2?.values[1] == 40)
        #expect(colLead2?.values[2] == nil)
        #expect(colLead2?.values[3] == nil)
        #expect(colLead2?.values[4] == nil)
    }

    @Test("castColumn throws partialCastFailure on invalid numeric string")
    func castColumnPartialFailure() throws {
        let nameCol = TypedColumn<String>(name: "str_val", values: ["10", "invalid_number", "30"])
        let df = try DataFrame(columns: [nameCol])
        #expect(throws: DataFrameError.self) {
            _ = try df.castColumn("str_val", to: Int64.self)
        }
    }

    @Test("addColumn computes column values from row closure")
    func addColumnRowClosure() throws {
        let df = try makeDF()
        let res = try df.addColumn("is_adult", as: Bool.self) { row in
            guard let age = row.value(column: "age", as: Int64.self) else { return nil }
            return age >= 18
        }
        #expect(res.columnNames.contains("is_adult"))
        let isAdults = res[column: "is_adult", as: Bool.self]?.values
        #expect(isAdults == [true, true, true, true, nil])
    }

    @Test("sample with ordered parameter preserves index sequence")
    func sampleOrdered() throws {
        let df = try makeDF()
        let sampledOrdered = df.sample(n: 3, seed: 42, ordered: true)
        #expect(sampledOrdered.shape.rows == 3)
    }
}
