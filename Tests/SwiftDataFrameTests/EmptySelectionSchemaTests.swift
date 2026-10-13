import Testing
import SwiftDataFrame

@Suite("Empty selections preserve schema")
struct EmptySelectionSchemaTests {
    private func frame() throws -> DataFrame {
        try DataFrame(columns: [
            TypedColumn<Int64>(name: "id", values: [7, nil, 9]),
            TypedColumn<Double>(name: "value", values: [nil, 2, 3]),
            TypedColumn<Bool>(name: "flag", values: [true, nil, false]),
            TypedColumn<String>(name: "label", values: ["a", nil, "c"])
        ])
    }

    @Test func gatheredResultRemainsUsable() throws {
        let source = try frame()
        let empty = source.gathered(at: [])
        #expect(empty.rowCount == 0)
        #expect(empty.columnNames == source.columnNames)
        #expect(empty[column: "id", as: Int64.self]?.values == [])
        #expect(empty[column: "value", as: Double.self]?.values == [])
        #expect(empty[column: "flag", as: Bool.self]?.values == [])
        #expect(empty[column: "label", as: String.self]?.values == [])
        for column in empty.columns { #expect(column.nullCount == 0) }
        let names = ["value", "flag", "id"]
        #expect(try empty.toFeatureMatrix(names).isEmpty)
        let prepared = try empty.prepareNumericBatch(names)
        #expect(prepared.rowCount == 0 && prepared.columnNames == names)
        #expect(prepared.originalRowIndices.isEmpty)
        #expect(source.rowCount == 3)
        #expect(source[column: "value", as: Double.self]?.values == [nil, 2, 3])
    }

    @Test(arguments: ["gather", "indices", "rows", "column", "head", "tail", "sample"])
    func emptySelectionPaths(operation: String) throws {
        let source = try frame()
        for input in [source, source.gathered(at: [])] {
            let output: DataFrame
            switch operation {
            case "gather": output = input.gathered(at: [])
            case "indices": output = input.filterRows { _ in false }
            case "rows": output = input.filter { _ in false }
            case "column": output = try input.filter(column: "value", where: .greaterThan(100.0))
            case "head": output = input.head(0)
            case "tail": output = input.tail(0)
            default: output = input.sample(n: 0, seed: 42)
            }
            #expect(output.rowCount == 0)
            #expect(output.columnNames == source.columnNames)
            #expect(output.columns.map(\.dtype) == source.columns.map(\.dtype))
            let selected = try output.select("label", "id")
            #expect(selected.columnNames == ["label", "id"])
        }
    }

    @Test func schemaFreeFramesAndColumnRemovalRemainSchemaFree() throws {
        let bare = DataFrame.empty
        let outputs = [bare.gathered(at: []), bare.filter { _ in false }, bare.filterRows { _ in false },
                       bare.head(0), bare.tail(0), bare.sample(n: 0)]
        for output in outputs {
            #expect(output.rowCount == 0)
            #expect(output.columnNames.isEmpty)
        }
        let source = try frame()
        #expect(try source.drop(source.columnNames).columnNames.isEmpty)
    }

    @Test func emptyPredicatesAreNotCalledAndNonemptyGatherIsUnchanged() throws {
        let empty = try frame().gathered(at: [])
        var calls = 0
        _ = empty.filterRows { _ in calls += 1; return true }
        _ = empty.filter { _ in calls += 1; return true }
        #expect(calls == 0)
        let source = try frame()
        let selected = source.gathered(at: [2, 0, 2])
        #expect(selected.columnNames == source.columnNames)
        #expect(selected[column: "id", as: Int64.self]?.values == [9, 7, 9])
    }
}
