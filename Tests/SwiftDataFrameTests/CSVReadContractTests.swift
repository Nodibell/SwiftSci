import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV ingestion contracts")
struct CSVReadContractTests {
    @Test("Stream inference remains scoped to each requested chunk")
    func chunkInference() async throws {
        let url = try write("value\n001\n002\n3.5\nNA\ntext\n5\n")
        defer { try? FileManager.default.removeItem(at: url) }
        let eager = try await DataFrame(csv: url)
        #expect(eager[column: "value", as: String.self]?.values == ["001", "002", "3.5", nil, "text", "5"])
        var frames = [DataFrame]()
        for try await frame in DataFrame.readCSVStream(contentsOf: url, chunkSize: 2) { frames.append(frame) }
        #expect(frames.count == 3)
        guard frames.count == 3 else { return }
        #expect(frames[0][column: "value", as: Int64.self]?.values == [1, 2])
        #expect(frames[1][column: "value", as: Double.self]?.values == [3.5, nil])
        #expect(frames[2][column: "value", as: String.self]?.values == ["text", "5"])
    }

    @Test("Reading, filtering, joining and both numerical preparation routes agree")
    func numericalWorkflow() async throws {
        let url = try write("id,power\n1,1.0\n2,2.5\n3,4.0\n")
        defer { try? FileManager.default.removeItem(at: url) }
        var options = CSVReadOptions()
        options.columnTypeOverrides = ["id": .int64, "power": .float64]
        let input = try await DataFrame(csv: url, options: options)
        let dimension = try DataFrame(columns: [
            TypedColumn<Int64>(name: "id", values: [2, 3]),
            TypedColumn<Double>(name: "weight", values: [10, 20])
        ])
        let selected = try input.filterFast(column: "power", where: .greaterThan(2))
        let joined = try selected.join(dimension, on: "id", how: .inner)
        let matrix = try joined.prepareNumericMatrix(["power", "weight"], missing: .reject)
        let compact = try joined.prepareNumericBatch(["power", "weight"])
        let packed = try compact.matrix(missing: .reject)
        #expect(matrix.values == [2.5, 10, 4, 20])
        #expect(packed.values == matrix.values)
        #expect(matrix.values[0] * matrix.values[1] + matrix.values[2] * matrix.values[3] == 105)
    }

    private func write(_ text: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".csv")
        try Data(text.utf8).write(to: url)
        return url
    }
}
