import Arrow
import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("Arrow date32 import", .serialized)
struct DataFrameArrowDate32Tests {
    private func table(_ chunks: [[Int32?]]) throws -> ArrowTable {
        let arrays = try chunks.map { days -> ArrowArray<Date> in
            let builder = try ArrowArrayBuilders.loadDate32ArrayBuilder()
            for day in days {
                builder.append(day.map { Date(timeIntervalSince1970: Double($0) * 86_400) })
            }
            return try builder.finish()
        }
        return ArrowTable.Builder().addColumn("date", chunked: try ChunkedArray(arrays)).finish()
    }

    @Test("Import a date32 column produced by Arrow's date builder")
    func builderRepresentation() throws {
        let input = try table([[0, 1, nil]])
        #expect(input.columns[0].type.id == .date32)
        let frame = try DataFrame(arrowTable: input)
        let values = try #require(frame[column: "date", as: Date.self])
        #expect(values.dtype == .date32)
        #expect(values.values.map { $0?.timeIntervalSince1970 } == [0, 86_400, nil])
    }

    @Test("Signed dates and the full day range survive chunked import",
          arguments: [ArrowNullStrategy.preserve, .zero, .nan])
    func signedRangeAndNulls(_ strategy: ArrowNullStrategy) throws {
        // 49,711 days crosses the UInt32 seconds range used by Arrow 21.0.0's
        // Date32Array subscript. Int32 extrema exercise the full physical type.
        let chunks: [[Int32?]] = [[], [Int32.min, -100_000, -1, nil], [], [0, 1, 49_710, 49_711, 100_000, Int32.max], []]
        let expected: [Double?] = [-185_542_587_187_200, -8_640_000_000, -86_400, nil,
                                  0, 86_400, 4_294_944_000, 4_295_030_400, 8_640_000_000, 185_542_587_100_800]
        let frame = try DataFrame(arrowTable: table(chunks), nullStrategy: strategy)
        let dates = try #require(frame[column: "date", as: Date.self])
        #expect(dates.values.map { $0?.timeIntervalSince1970 } == expected)
        #expect(dates.nullCount == 1)
        #expect(frame.shape.rows == expected.count)
    }

    @Test("Empty and all-null date32 columns preserve their type")
    func emptyAndMissing() throws {
        let empty = try DataFrame(arrowTable: table([[], [], []]))
        #expect(empty[column: "date", as: Date.self]?.values == [])
        #expect(empty.shape.rows == 0)
        let missing = try DataFrame(arrowTable: table([[], [nil, nil], []]))
        #expect(missing[column: "date", as: Date.self]?.values == [nil, nil])
        #expect(missing[column: "date"]?.nullCount == 2)
    }

    @Test("Imported dates outlive the source without aliasing its storage")
    func ownedDates() throws {
        var input: ArrowTable? = try table([[-1, nil], [100_000]])
        weak var buffer: ArrowBuffer?
        let frame = try DataFrame(arrowTable: input!)
        do {
            let chunks: ChunkedArray<Date> = input!.columns[0].data()
            buffer = chunks.arrays[0].arrowData.buffers[1]
            buffer!.rawPointer.storeBytes(of: Int32(4), as: Int32.self)
        }
        input = nil
        #expect(buffer == nil)
        #expect(frame[column: "date", as: Date.self]?.values.map { $0?.timeIntervalSince1970 } == [-86_400, nil, 8_640_000_000])
    }
}
