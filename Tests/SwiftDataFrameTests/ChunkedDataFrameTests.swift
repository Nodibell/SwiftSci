import XCTest
@testable import SwiftDataFrame

final class ChunkedDataFrameTests: XCTestCase {

    func testChunkedDataFrameBasicIterationAndCollect() async throws {
        let col1A = TypedColumn<Int64>(name: "id", values: [1, 2, 3])
        let col2A = TypedColumn<String>(name: "name", values: ["A", "B", "C"])
        let df1 = try DataFrame(columns: [col1A, col2A])

        let col1B = TypedColumn<Int64>(name: "id", values: [4, 5])
        let col2B = TypedColumn<String>(name: "name", values: ["D", "E"])
        let df2 = try DataFrame(columns: [col1B, col2B])

        let chunked = ChunkedDataFrame(chunks: [df1, df2])
        let totalRows = try await chunked.rowCount()
        XCTAssertEqual(totalRows, 5)

        let collected = try await chunked.collect()
        XCTAssertEqual(collected.rowCount, 5)
        XCTAssertEqual(collected.columnNames, ["id", "name"])

        let idCol = collected[column: "id"] as? TypedColumn<Int64>
        XCTAssertEqual(idCol?.values, [1, 2, 3, 4, 5])
    }

    func testChunkedDataFrameFilterAndSelect() async throws {
        let col1 = TypedColumn<Int64>(name: "id", values: [10, 20, 30, 40, 50])
        let col2 = TypedColumn<Double>(name: "score", values: [1.5, 2.5, 3.5, 4.5, 5.5])
        let col3 = TypedColumn<String>(name: "tag", values: ["x", "y", "x", "y", "z"])
        let df = try DataFrame(columns: [col1, col2, col3])

        let chunked = df.chunked(by: 2)

        let filtered = chunked.filter { row in
            guard let score = row.double("score") else { return false }
            return score > 3.0
        }

        let selected = filtered.select("id", "score")
        let result = try await selected.collect()

        XCTAssertEqual(result.rowCount, 3)
        XCTAssertEqual(result.columnNames, ["id", "score"])

        let ids = (result[column: "id"] as? TypedColumn<Int64>)?.values
        XCTAssertEqual(ids, [30, 40, 50])
    }

    func testDataFrameConcat() throws {
        let df1 = try DataFrame(columns: [
            TypedColumn<Int64>(name: "a", values: [1, 2]),
            TypedColumn<Double>(name: "b", values: [1.1, 2.2])
        ])
        let df2 = try DataFrame(columns: [
            TypedColumn<Int64>(name: "a", values: [3]),
            TypedColumn<Double>(name: "b", values: [3.3])
        ])
        let df3 = try DataFrame(columns: [
            TypedColumn<Int64>(name: "a", values: [4, 5]),
            TypedColumn<Double>(name: "b", values: [4.4, 5.5])
        ])

        let combined = try DataFrame.concat([df1, df2, df3])
        XCTAssertEqual(combined.rowCount, 5)
        let aVals = (combined[column: "a"] as? TypedColumn<Int64>)?.values
        XCTAssertEqual(aVals, [1, 2, 3, 4, 5])
    }

    func testMemoryMappedReaderLinePartitioning() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent(UUID().uuidString + ".txt")

        let content = "line1,100\nline2,200\nline3,300\nline4,400\nline5,500\n"
        try content.write(to: tempFile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let mmapReader = try MemoryMappedReader(url: tempFile)
        XCTAssertEqual(mmapReader.size, content.utf8.count)

        let partitions = mmapReader.partitionByLines(targetChunkByteSize: 15)
        XCTAssertGreaterThanOrEqual(partitions.count, 1)

        var assembled = ""
        for part in partitions {
            let partData = mmapReader.data(in: part)
            assembled += String(decoding: partData, as: UTF8.self)
        }
        XCTAssertEqual(assembled, content)
    }
}
