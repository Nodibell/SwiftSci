import Foundation
import XCTest
@testable import SwiftDataFrame

final class ParquetInteropTests: XCTestCase {
    private func fixture(_ name: String) throws -> URL {
        try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "parquet", subdirectory: "ParquetInterop"))
    }

    func testExternalPackedBooleansWithNullsAndPartialByte() async throws {
        let frame = try await ParquetReader.read(url: fixture("pyarrow-25.0.1"))
        XCTAssertEqual(frame.rowCount, 129)
        let flags = try XCTUnwrap(frame[column: "flag"] as? TypedColumn<Bool>)
        let labels = try XCTUnwrap(frame[column: "label"] as? TypedColumn<String>)
        let ids = try XCTUnwrap(frame[column: "id"] as? TypedColumn<Int64>)
        for i in 0..<129 {
            XCTAssertEqual(ids.values[i], Int64(i))
            XCTAssertEqual(flags.values[i], i % 5 == 0 ? nil : i % 2 == 0)
            XCTAssertEqual(labels.values[i], i % 7 == 0 ? nil : "row\(i)")
        }
    }

    func testLegacyWholePageCompressionRemainsReadable() async throws {
        let frame = try await ParquetReader.read(url: fixture("legacy-swiftsci"))
        XCTAssertEqual(frame.rowCount, 129)
        let flags = try XCTUnwrap(frame[column: "flag"] as? TypedColumn<Bool>)
        for i in 0..<129 { XCTAssertEqual(flags.values[i], i % 2 == 0) }
    }

    func testNewWriterPreservesExternalNullableValues() async throws {
        let original = try await ParquetReader.read(url: fixture("pyarrow-25.0.1"))
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".parquet")
        defer { try? FileManager.default.removeItem(at: path) }
        try await ParquetWriter.write(dataFrame: original, to: path)
        let frame = try await ParquetReader.read(url: path)
        XCTAssertEqual((frame[column: "flag"] as? TypedColumn<Bool>)?.values,
                       (original[column: "flag"] as? TypedColumn<Bool>)?.values)
        XCTAssertEqual((frame[column: "label"] as? TypedColumn<String>)?.values,
                       (original[column: "label"] as? TypedColumn<String>)?.values)
    }
}
