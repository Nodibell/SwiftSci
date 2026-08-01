import Testing
import Foundation
@testable import SwiftDataFrame

@Suite("LazyDataFrame Query Optimization Tests")
struct LazyDataFrameTests {
    
    @Test("Eager DataFrame lazy collection")
    func testLazyEagerCollection() async throws {
        let nameCol = TypedColumn<String>(name: "name", values: ["Alice", "Bob", "Charlie", "David"])
        let scoreCol = TypedColumn<Double>(name: "score", values: [75.0, 92.0, 88.0, 60.0])
        let ageCol = TypedColumn<Int32>(name: "age", values: [25, 30, 35, 40])
        
        let df = try DataFrame(columns: [nameCol, scoreCol, ageCol])
        
        let result = try await df.lazy()
            .filter { row in (row.double("score") ?? 0) >= 80.0 }
            .filter { row in (row["age", as: Int32.self] ?? 0) <= 32 }
            .select("name", "score")
            .collect()
        
        #expect(result.shape.rows == 1)
        #expect(result.shape.columns == 2)
        #expect(result.columnNames == ["name", "score"])
        
        let names = result[column: "name", as: String.self]?.values
        #expect(names == ["Bob"])
    }
    
    @Test("QueryPlan optimization merges consecutive filters")
    func testQueryPlanFilterMerging() {
        let f1: @Sendable (DataFrameRow) -> Bool = { _ in true }
        let f2: @Sendable (DataFrameRow) -> Bool = { _ in false }
        
        let plan = QueryPlan(nodes: [
            .filter(predicate: f1),
            .filter(predicate: f2),
            .select(columns: ["a"])
        ])
        
        let optimized = plan.optimized()
        #expect(optimized.nodes.count == 2) // 1 merged filter + 1 select
    }
    
    @Test("Lazy Feather source reading")
    func testLazyFeatherReading() async throws {
        let colA = TypedColumn<Int32>(name: "A", values: [10, 20, 30])
        let colB = TypedColumn<String>(name: "B", values: ["X", "Y", "Z"])
        let df = try DataFrame(columns: [colA, colB])
        
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("lazy_\(UUID().uuidString).feather")
        defer { try? FileManager.default.removeItem(at: tempFile) }
        
        try await df.writeFeather(to: tempFile)
        
        let lazyResult = try await DataFrame.lazyFeather(url: tempFile)
            .filter { row in (row["A", as: Int32.self] ?? 0) > 15 }
            .collect()
        
        #expect(lazyResult.shape.rows == 2)
        let colAVals = lazyResult[column: "A", as: Int32.self]?.values
        #expect(colAVals == [20, 30])
    }
}
