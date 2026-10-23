import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV storage choices during inference retries")
struct CSVDoubleStorageInferenceTests {
    @Test("Integer promotion, text rejection and explicit coercion retain the selected layout",
          arguments: [CSVColumnConversionMode.serial, .parallel])
    func retries(_ conversion: CSVColumnConversionMode) async throws {
        let good = FileManager.default.temporaryDirectory.appendingPathComponent("csv-promotion-\(UUID().uuidString).csv")
        let bad = FileManager.default.temporaryDirectory.appendingPathComponent("csv-rejection-\(UUID().uuidString).csv")
        defer { try? FileManager.default.removeItem(at: good); try? FileManager.default.removeItem(at: bad) }
        let csv = "x\n" + String(repeating: "42\n", count: 1_001) + "1.5\n"
        try Data(csv.utf8).write(to: good)
        try Data((csv + "invalid\n").utf8).write(to: bad)
        for storage in [CSVDoubleStorage.optional, .compact] {
            var options = CSVReadOptions()
            options.doubleStorage = storage
            options.columnConversionMode = conversion
            let promoted = try await DataFrame(csv: good, options: options)
            let double = try #require(promoted[column: "x", as: Double.self])
            #expect(double.count == 1_002 && double.nullCount == 0)
            #expect(double[0] == 42 && double[1_001] == 1.5)
            #expect((double.compactNumericStorage != nil) == (storage == .compact))
            let rejected = try await DataFrame(csv: bad, options: options)
            let text = try #require(rejected[column: "x", as: String.self])
            #expect(text.count == 1_003 && text.nullCount == 0)
            #expect(text[0] == "42" && text[1_001] == "1.5" && text[1_002] == "invalid")
            options.columnTypeOverrides = ["x": .float64]
            let forced = try await DataFrame(csv: bad, options: options)
            let coerced = try #require(forced[column: "x", as: Double.self])
            #expect(coerced.count == 1_003 && coerced.nullCount == 1)
            #expect(coerced[1_001] == 1.5 && coerced[1_002] == nil)
            #expect((coerced.compactNumericStorage != nil) == (storage == .compact))
        }
    }
}
