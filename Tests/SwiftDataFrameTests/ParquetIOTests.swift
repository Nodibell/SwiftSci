import XCTest
@testable import SwiftDataFrame

final class ParquetIOTests: XCTestCase {

    func testSnappyCompressionAndDecompressionRoundTrip() throws {
        let originalText = "SwiftSci v3.5.0 High-Performance Out-of-Core Parquet Processing with Pure Swift! " +
            String(repeating: "Repeating pattern for LZ77 compression testing. ", count: 50)
        let originalData = Data(originalText.utf8)

        let compressed = SnappyDecompressor.compress(data: originalData)
        XCTAssertLessThan(compressed.count, originalData.count)

        let decompressed = try SnappyDecompressor.decompress(data: compressed)
        XCTAssertEqual(decompressed, originalData)
        XCTAssertEqual(String(decoding: decompressed, as: UTF8.self), originalText)
    }

    func testParquetRoundTripSerialization() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let parquetURL = tempDir.appendingPathComponent(UUID().uuidString + ".parquet")
        defer { try? FileManager.default.removeItem(at: parquetURL) }

        let idCol = TypedColumn<Int64>(name: "id", values: [1, 2, nil, 4, 5])
        let int32Col = TypedColumn<Int32>(name: "age", values: [20, 25, 30, nil, 40])
        let scoreCol = TypedColumn<Double>(name: "score", values: [99.5, nil, 88.0, 77.25, 66.0])
        let activeCol = TypedColumn<Bool>(name: "active", values: [true, false, true, nil, false])
        let cityCol = TypedColumn<String>(name: "city", values: ["Kyiv", "Lviv", nil, "Odesa", "Kharkiv"])

        let originalDF = try DataFrame(columns: [idCol, int32Col, scoreCol, activeCol, cityCol])

        // Write to Parquet
        try await originalDF.writeParquet(to: parquetURL)

        XCTAssertTrue(FileManager.default.fileExists(atPath: parquetURL.path))

        // Read from Parquet
        let loadedDF = try await DataFrame(parquet: parquetURL)

        XCTAssertEqual(loadedDF.rowCount, 5)
        XCTAssertEqual(loadedDF.columnNames, ["id", "age", "score", "active", "city"])

        let readIdCol = loadedDF[column: "id"] as? TypedColumn<Int64>
        XCTAssertEqual(readIdCol?.values, [1, 2, nil, 4, 5])

        let readAgeCol = loadedDF[column: "age"] as? TypedColumn<Int32>
        XCTAssertEqual(readAgeCol?.values, [20, 25, 30, nil, 40])

        let readScoreCol = loadedDF[column: "score"] as? TypedColumn<Double>
        XCTAssertEqual(readScoreCol?.values, [99.5, nil, 88.0, 77.25, 66.0])

        let readActiveCol = loadedDF[column: "active"] as? TypedColumn<Bool>
        XCTAssertEqual(readActiveCol?.values, [true, false, true, nil, false])

        let readCityCol = loadedDF[column: "city"] as? TypedColumn<String>
        XCTAssertEqual(readCityCol?.values, ["Kyiv", "Lviv", nil, "Odesa", "Kharkiv"])
    }

    func testLazyParquetPipeline() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let parquetURL = tempDir.appendingPathComponent(UUID().uuidString + ".parquet")
        defer { try? FileManager.default.removeItem(at: parquetURL) }

        let df = try DataFrame(columns: [
            TypedColumn<Int64>(name: "id", values: [1, 2, 3, 4, 5]),
            TypedColumn<Double>(name: "val", values: [10.0, 20.0, 30.0, 40.0, 50.0])
        ])
        try await df.writeParquet(to: parquetURL)

        let lazyDF = DataFrame.lazyParquet(url: parquetURL)
            .filter { row in
                guard let val = row.double("val") else { return false }
                return val >= 30.0
            }
            .select("id")

        let collected = try await lazyDF.collect()
        XCTAssertEqual(collected.rowCount, 3)
        XCTAssertEqual(collected.columnNames, ["id"])

        let ids = (collected[column: "id"] as? TypedColumn<Int64>)?.values
        XCTAssertEqual(ids, [3, 4, 5])
    }

    func testSnappySliceFromGoEmotions() throws {
        let fileURL = URL(fileURLWithPath: "/Users/oleksiichumak/.gemini/antigravity-ide/brain/97383ad2-c472-4b55-8e5e-3dbffc712586/scratch/go_emotions_test.parquet")
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return }
        let data = try Data(contentsOf: fileURL)
        let snappySlice = data.subdata(in: 23 ..< (23 + 50922))
        let decompressed = try SnappyDecompressor.decompress(data: snappySlice)
        print("✅ SnappyDecompressor decompressed: \(decompressed.count) bytes!")
        XCTAssertEqual(decompressed.count, 72042)
    }

    func testReadGoEmotionsHuggingFaceParquet() async throws {
        let fileURL = URL(fileURLWithPath: "/Users/oleksiichumak/.gemini/antigravity-ide/brain/97383ad2-c472-4b55-8e5e-3dbffc712586/scratch/go_emotions_test.parquet")
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return }
        let df = try await DataFrame(parquet: fileURL)
        print("✅ GoEmotions Parquet Loaded! Rows: \(df.rowCount), Columns: \(df.columnNames)")
        XCTAssertEqual(df.rowCount, 5427)
        XCTAssertTrue(df.columnNames.contains("text"))
        XCTAssertTrue(df.columnNames.contains("labels"))
        XCTAssertTrue(df.columnNames.contains("id"))

        // Inspect sample values
        let textCol = df[column: "text"] as? TypedColumn<String>
        XCTAssertNotNil(textCol)
        XCTAssertEqual(textCol?.values.first, "I’m really sorry about your situation :( Although I love the names Sapphira, Cirilla, and Scarlett!")

        let idCol = df[column: "id"] as? TypedColumn<String>
        XCTAssertNotNil(idCol)
        XCTAssertEqual(idCol?.values.first, "eecwqtt")

        let labelsCol = df[column: "labels"] as? TypedColumn<String>
        XCTAssertNotNil(labelsCol)
        XCTAssertEqual(labelsCol?.values.first, "[25]")
    }
}
