import Testing
import SwiftDataFrame

@Suite("DataFrame GroupBy & Aggregations")
struct DataFrameGroupByTests {

    private func makeDF() throws -> DataFrame {
        let category = TypedColumn<String>(name: "category", values: ["A", "A", "B", "B", "A"])
        let val1     = TypedColumn<Int64>(name: "val1", values: [10, 20, 30, 40, 50])
        let val2     = TypedColumn<Double>(name: "val2", values: [1.5, 2.5, 3.5, 4.5, 5.5])
        return try DataFrame(columns: [category, val1, val2])
    }

    @Test("groupBy count is correct")
    func testGroupByCount() throws {
        let df = try makeDF()
        let grouped = df.groupBy("category").count()
        
        #expect(grouped.shape.rows == 2)
        #expect(grouped.shape.columns == 3) // category, val1, val2 (holding counts)

        // Find category A
        let categories = grouped[column: "category", as: String.self]?.values
        #expect(categories?.contains("A") == true)
        #expect(categories?.contains("B") == true)

        for i in 0..<grouped.shape.rows {
            let cat = grouped[column: "category"]?.value(at: i) as? String
            let val1Count = grouped[column: "val1"]?.value(at: i) as? Int64
            let val2Count = grouped[column: "val2"]?.value(at: i) as? Int64
            
            if cat == "A" {
                #expect(val1Count == 3)
                #expect(val2Count == 3)
            } else if cat == "B" {
                #expect(val1Count == 2)
                #expect(val2Count == 2)
            }
        }
    }

    @Test("groupBy sum is correct")
    func testGroupBySum() throws {
        let df = try makeDF()
        let grouped = df.groupBy("category").sum()

        for i in 0..<grouped.shape.rows {
            let cat = grouped[column: "category"]?.value(at: i) as? String
            let val1Sum = grouped[column: "val1"]?.value(at: i) as? Double
            let val2Sum = grouped[column: "val2"]?.value(at: i) as? Double
            
            if cat == "A" {
                #expect(val1Sum == 80.0) // 10 + 20 + 50
                #expect(val2Sum == 9.5)  // 1.5 + 2.5 + 5.5
            } else if cat == "B" {
                #expect(val1Sum == 70.0) // 30 + 40
                #expect(val2Sum == 8.0)  // 3.5 + 4.5
            }
        }
    }

    @Test("groupBy mean is correct")
    func testGroupByMean() throws {
        let df = try makeDF()
        let grouped = df.groupBy("category").mean()

        for i in 0..<grouped.shape.rows {
            let cat = grouped[column: "category"]?.value(at: i) as? String
            let val1Mean = grouped[column: "val1"]?.value(at: i) as? Double
            let val2Mean = grouped[column: "val2"]?.value(at: i) as? Double
            
            if cat == "A" {
                #expect(abs((val1Mean ?? 0) - 26.66666667) < 1e-5) // 80 / 3
                #expect(abs((val2Mean ?? 0) - 3.16666667) < 1e-5)  // 9.5 / 3
            } else if cat == "B" {
                #expect(val1Mean == 35.0) // 70 / 2
                #expect(val2Mean == 4.0)  // 8 / 2
            }
        }
    }

    @Test("groupBy agg applies multiple custom aggregations")
    func testGroupByAgg() throws {
        let df = try makeDF()
        let grouped = df.groupBy("category").agg([
            "val1": .max,
            "val2": .min
        ])

        #expect(grouped.columnNames.contains("val1_max"))
        #expect(grouped.columnNames.contains("val2_min"))

        for i in 0..<grouped.shape.rows {
            let cat = grouped[column: "category"]?.value(at: i) as? String
            let val1Max = grouped[column: "val1_max"]?.value(at: i) as? Double
            let val2Min = grouped[column: "val2_min"]?.value(at: i) as? Double
            
            if cat == "A" {
                #expect(val1Max == 50.0)
                #expect(val2Min == 1.5)
            } else if cat == "B" {
                #expect(val1Max == 40.0)
                #expect(val2Min == 3.5)
            }
        }
    }
}
