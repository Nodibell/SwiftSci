import Testing
import SwiftDataFrame

@Suite("Int64 row gathering")
struct Int64GatherTests {
    @Test("Gather preserves exact integers, nulls, duplicates and requested order")
    func exactIntegers() throws {
        let values: [Int64?] = [.min, 9_007_199_254_740_993, nil, .max, -9_007_199_254_740_993]
        let column: any AnyColumn = TypedColumn<Int64>(name: "id", values: values)
        let indices = [3, 2, 1, 0, 4, 2, 1]
        let result = try #require(column.gathered(at: indices) as? TypedColumn<Int64>)
        #expect(result.values == indices.map { values[$0] })
        #expect(result.nullCount == 2)
        #expect(result.name == "id")
        #expect(result.dtype == column.dtype)
        let empty = try #require(column.gathered(at: []) as? TypedColumn<Int64>)
        #expect(empty.values.isEmpty)
        #expect(empty.nullCount == 0)
        #expect((column as? TypedColumn<Int64>)?.values == values)
    }

    @Test("Filtering preserves nullable integer columns at different selectivities", arguments: [0, 33, 13_001])
    func filterMatrix(count: Int) throws {
        for nullEvery in [0, 1, 2, 101] {
            let scores: [Double?] = (0..<count).map {
                nullEvery > 0 && $0 % nullEvery == 0 ? nil : Double($0 % 100)
            }
            let ids: [Int64?] = (0..<count).map {
                $0 % 7 == 0 ? nil : 9_007_199_254_740_993 + Int64($0)
            }
            let positions = (0..<count).map { Int64($0) }
            let frame = try DataFrame(columns: [
                TypedColumn<Double>(name: "score", values: scores),
                TypedColumn<Int64>(name: "id", values: ids),
                TypedColumn<Int64>(name: "position", values: positions),
                TypedColumn<Int64>(name: "other", values: positions)
            ])
            for threshold in [-1.0, 50.0, 101.0] {
                let indices = scores.indices.filter { (scores[$0] ?? -.infinity) > threshold }
                let expectedIDs = indices.map { ids[$0] }
                for result in [
                    try frame.filter(column: "score", where: .greaterThan(threshold)),
                    try frame.filterFast(column: "score", where: .greaterThan(threshold))
                ] {
                    #expect(result.rowCount == indices.count)
                    if !indices.isEmpty {
                        let selected = try #require(result[column: "id", as: Int64.self])
                        #expect(selected.values == expectedIDs)
                        #expect(selected.nullCount == expectedIDs.filter { $0 == nil }.count)
                        #expect(result[column: "position", as: Int64.self]?.values == indices.map { Int64($0) })
                        #expect(result[column: "score", as: Double.self]?.values == indices.map { scores[$0] })
                        #expect(result[column: "other", as: Int64.self]?.values == indices.map { Int64($0) })
                    }
                }
            }
            #expect(frame[column: "id", as: Int64.self]?.values == ids)
        }
    }
}
