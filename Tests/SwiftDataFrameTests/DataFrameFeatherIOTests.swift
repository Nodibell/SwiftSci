import Testing
import Foundation
@testable import SwiftDataFrame

@Suite("DataFrame Feather / Arrow IPC I/O Tests")
struct DataFrameFeatherIOTests {
    
    @Test("Round-trip Feather file serialization and deserialization")
    func testFeatherFileRoundTrip() async throws {
        let ints = TypedColumn<Int32>(name: "id", values: [1, 2, nil, 4])
        let doubles = TypedColumn<Double>(name: "score", values: [98.5, 87.0, 92.3, nil])
        let strings = TypedColumn<String>(name: "name", values: ["Alpha", "Beta", "Gamma", nil])
        let bools = TypedColumn<Bool>(name: "active", values: [true, false, true, false])
        
        let originalDF = try DataFrame(columns: [ints, doubles, strings, bools])
        
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_\(UUID().uuidString).feather")
        defer { try? FileManager.default.removeItem(at: tempFile) }
        
        try await originalDF.writeFeather(to: tempFile)
        
        #expect(FileManager.default.fileExists(atPath: tempFile.path))
        
        let loadedDF = try await DataFrame(feather: tempFile)
        
        #expect(loadedDF.shape.rows == 4)
        #expect(loadedDF.shape.columns == 4)
        
        let loadedInts = loadedDF[column: "id", as: Int32.self]?.values
        #expect(loadedInts == [1, 2, nil, 4])
        
        let loadedDoubles = loadedDF[column: "score", as: Double.self]?.values
        #expect(loadedDoubles == [98.5, 87.0, 92.3, nil])
        
        let loadedStrings = loadedDF[column: "name", as: String.self]?.values
        #expect(loadedStrings == ["Alpha", "Beta", "Gamma", nil])
        
        let loadedBools = loadedDF[column: "active", as: Bool.self]?.values
        #expect(loadedBools == [true, false, true, false])
    }
    
    @Test("Round-trip Feather Data buffer serialization")
    func testFeatherDataRoundTrip() async throws {
        let col = TypedColumn<Double>(name: "val", values: [1.1, 2.2, 3.3])
        let df = try DataFrame(columns: [col])
        
        let featherData = try df.writeFeatherData()
        #expect(!featherData.isEmpty)
        
        let loadedDF = try await FeatherReader.read(data: featherData)
        #expect(loadedDF.shape.rows == 3)
        #expect(loadedDF.shape.columns == 1)
        
        let vals = loadedDF[column: "val", as: Double.self]?.values
        #expect(vals == [1.1, 2.2, 3.3])
    }
}
