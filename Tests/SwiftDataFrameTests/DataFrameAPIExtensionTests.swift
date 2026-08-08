import Testing
import Foundation
import SwiftDataFrame

@Suite("DataFrame API Extension Tests (Phase 2)")
struct DataFrameAPIExtensionTests {

    @Test("mapColumn transforms values functional-style")
    func testMapColumn() throws {
        let priceCol = TypedColumn<Double>(name: "price", values: [10.0, 20.0, nil])
        let df = try DataFrame(columns: [priceCol])
        
        let df2 = try df.mapColumn("price", as: Double.self) { val in
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
        #expect(matchedRow["city"] as? String == "Lviv")
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
