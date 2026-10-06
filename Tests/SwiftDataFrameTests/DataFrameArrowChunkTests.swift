import Arrow
import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("Chunked Arrow imports", .serialized)
struct DataFrameArrowChunkTests {
    private func column<T>(_ name: String, _ chunks: [[T?]]) throws -> ArrowColumn {
        let arrays = try chunks.map { values -> ArrowArray<T> in
            let builder = try ArrowArrayBuilders.loadBuilder(T.self)
            for value in values { builder.appendAny(value) }
            return try #require(builder.toHolder().array as? ArrowArray<T>)
        }
        return ArrowTable.Builder().addColumn(name, chunked: try ChunkedArray(arrays)).finish().columns[0]
    }

    @Test("Empty chunks before, between, and after values preserve row order")
    func emptyChunks() throws {
        let col = try column("x", [[Double?](), [1, nil], [], [3, 4], []])
        let frame = try DataFrame(arrowTable: ArrowTable.Builder().addColumn(col).finish())
        #expect(frame.shape.rows == 4)
        #expect(frame[column: "x", as: Double.self]?.values == [1, nil, 3, 4])
        #expect(frame[column: "x"]?.nullCount == 1)
    }

    @Test("All-empty chunks preserve the typed column")
    func allEmptyChunks() throws {
        let col = try column("x", [[Int64?](), [], []])
        let frame = try DataFrame(arrowTable: ArrowTable.Builder().addColumn(col).finish())
        #expect(frame.shape.rows == 0)
        #expect(frame[column: "x", as: Int64.self]?.values == [])
    }

    @Test("Different chunk boundaries preserve null strategies and exact values",
          arguments: [ArrowNullStrategy.preserve, .zero, .nan])
    func nullStrategies(_ strategy: ArrowNullStrategy) throws {
        let table = ArrowTable.Builder()
        _ = try table.addColumn(column("i32", [[Int32.min], [], [nil, 0, Int32.max]]))
        _ = try table.addColumn(column("i64", [[Int64.min, nil], [9_007_199_254_740_993, Int64.max]]))
        _ = try table.addColumn(column("f32", [[Float.nan, nil, -0.0], [], [.infinity]]))
        _ = try table.addColumn(column("f64", [[], [Double.nan], [nil, -0.0, -.infinity]]))
        _ = try table.addColumn(column("bool", [[true, nil], [], [false, true]]))
        _ = try table.addColumn(column("str", [["α", nil, ""], ["👩🏽‍🔬"]]))
        let frame = try DataFrame(arrowTable: table.finish(), nullStrategy: strategy)
        #expect(frame.shape.rows == 4)
        #expect(frame.shape.columns == 6)
        #expect(frame[column: "i32", as: Int32.self]?.values == [Int32.min, strategy == .zero ? 0 : nil, 0, Int32.max])
        #expect(frame[column: "i64", as: Int64.self]?.values == [Int64.min, strategy == .zero ? 0 : nil, 9_007_199_254_740_993, Int64.max])
        #expect(frame[column: "bool", as: Bool.self]?.values == [true, strategy == .zero ? false : nil, false, true])
        #expect(frame[column: "str", as: String.self]?.values == ["α", nil, "", "👩🏽‍🔬"])
        let floats = try #require(frame[column: "f32", as: Float.self])
        let doubles = try #require(frame[column: "f64", as: Double.self])
        #expect(floats.values[0]?.isNaN == true)
        #expect(doubles.values[0]?.isNaN == true)
        #expect(floats.values[2]?.sign == .minus)
        #expect(doubles.values[2]?.sign == .minus)
        #expect(floats.values[3] == .infinity)
        #expect(doubles.values[3] == -.infinity)
        switch strategy {
        case .preserve:
            #expect(floats.values[1] == nil && doubles.values[1] == nil)
        case .zero:
            #expect(floats.values[1] == 0 && doubles.values[1] == 0)
        case .nan:
            #expect(floats.values[1]?.isNaN == true && doubles.values[1]?.isNaN == true)
        }
        #expect(floats.nullCount == (strategy == .preserve ? 1 : 0))
        #expect(doubles.nullCount == (strategy == .preserve ? 1 : 0))
    }

    @Test("Import owns its values independently of the producer")
    func ownedSnapshot() throws {
        var col: ArrowColumn? = try column("x", [[1.0, nil], [3.0]])
        var table: ArrowTable? = ArrowTable.Builder().addColumn(col!).finish()
        let frame = try DataFrame(arrowTable: table!)
        do {
            let chunks: ChunkedArray<Double> = col!.data()
            chunks.arrays[0].arrowData.buffers[1].rawPointer.storeBytes(of: 99.0, as: Double.self)
        }
        table = nil
        col = nil
        #expect(frame[column: "x", as: Double.self]?.values == [1, nil, 3])
    }
}
