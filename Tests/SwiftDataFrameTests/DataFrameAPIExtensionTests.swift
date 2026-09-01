import Testing
import Foundation
import SwiftDataFrame

@Suite("DataFrame API Extension Tests (Phase 2)")
struct DataFrameAPIExtensionTests {

    @Test("mapColumn transforms values functional-style")
    func testMapColumn() throws {
        let priceCol = TypedColumn<Double>(name: "price", values: [10.0, 20.0, nil])
        let df = try DataFrame(columns: [priceCol])
        
        let df2 = try df.mapColumn("price", as: Double.self) { (val: Double?) -> Double? in
            val.map { $0 * 1.2 }
        }
        
        guard let newCol = df2[column: "price", as: Double.self] else {
            Issue.record("Column not found")
            return
        }
        
        #expect(newCol[0] == 12.0)
        #expect(newCol[1] == 24.0)
        #expect(newCol[2] == nil)
    }

    @Test("DataFrameRow typed subscript and helpers access values safely")
    func testDataFrameRowTypedAccessors() throws {
        let ageCol = TypedColumn<Int64>(name: "age", values: [25, 35])
        let cityCol = TypedColumn<String>(name: "city", values: ["Kyiv", "Lviv"])
        let scoreCol = TypedColumn<Double>(name: "score", values: [95.5, 88.0])
        let df = try DataFrame(columns: [ageCol, cityCol, scoreCol])
        
        let filtered = df.filter { row in
            (row.int("age") ?? 0) > 30 && row.string("city") == "Lviv" && (row.double("score") ?? 0) < 90.0
        }
        
        #expect(filtered.shape.rows == 1)
        let matchedRow = filtered.row(at: 0)
        #expect(matchedRow.string("city") == "Lviv")
    }

    @Test("GroupedDataFrame.transform expands aggregated results back to original rows")
    func testGroupedDataFrameTransform() throws {
        let deptCol = TypedColumn<String>(name: "dept", values: ["Engineering", "Engineering", "Sales", "Sales"])
        let salaryCol = TypedColumn<Double>(name: "salary", values: [100.0, 200.0, 50.0, 150.0])
        let df = try DataFrame(columns: [deptCol, salaryCol])
        
        let res = df.groupBy("dept").transform(["salary": .mean])
        
        #expect(res.shape.rows == 4)
        guard let meanCol = res[column: "salary_group_mean", as: Double.self] else {
            Issue.record("Aggregated transform column missing")
            return
        }
        
        // Engineering mean = 150.0, Sales mean = 100.0
        #expect(meanCol[0] == 150.0)
        #expect(meanCol[1] == 150.0)
        #expect(meanCol[2] == 100.0)
        #expect(meanCol[3] == 100.0)
    }

    @Test("TypedColumn.unique and DataFrame.unique return deduplicated elements and rows")
    func testUniqueProperty() throws {
        // 1. Column unique
        let col = TypedColumn<String>(name: "category", values: ["спорт", "новини", "спорт", "політика", "новини", nil, nil])
        let colUnique = col.typedUnique
        #expect(colUnique.values == ["спорт", "новини", "політика", nil])
        
        let anyColUnique = col.unique
        #expect(anyColUnique.count == 4)

        // 2. DataFrame unique
        let c1 = TypedColumn<String>(name: "cat", values: ["A", "B", "A", "B", "C"])
        let c2 = TypedColumn<Int64>(name: "val", values: [1, 2, 1, 3, 1])
        let df = try DataFrame(columns: [c1, c2])
        
        let dfUnique = df.unique
        #expect(dfUnique.shape.rows == 4) // (A,1), (B,2), (B,3), (C,1)
        #expect(dfUnique[column: "cat", as: String.self]?.values == ["A", "B", "B", "C"])
        #expect(dfUnique[column: "val", as: Int64.self]?.values == [1, 2, 3, 1])

        // Edge case: empty DataFrame.unique returns itself (guard branch)
        let empty = try DataFrame(columns: [TypedColumn<String>(name: "x", values: [])])
        #expect(empty.unique.shape.rows == 0)
    }

    @Test("TypedColumn sortedIndices for Float, Int32, and Date types")
    func testTypedColumnSortedIndicesSpecializedTypes() throws {
        let floatCol = TypedColumn<Float>(name: "float", values: [3.0, 1.0, 2.0])
        #expect(floatCol.sortedIndices(ascending: true) == [1, 2, 0])

        let int32Col = TypedColumn<Int32>(name: "int32", values: [30, 10, 20])
        #expect(int32Col.sortedIndices(ascending: true) == [1, 2, 0])

        let now = Date()
        let date1 = now.addingTimeInterval(10)
        let date2 = now.addingTimeInterval(20)
        let dateCol = TypedColumn<Date>(name: "date", values: [date2, now, date1])
        #expect(dateCol.sortedIndices(ascending: true) == [1, 2, 0])
    }
}
