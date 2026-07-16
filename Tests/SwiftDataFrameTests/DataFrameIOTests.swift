import Testing
import Foundation
@testable import SwiftDataFrame

@Suite("DataFrame I/O Tests")
struct DataFrameIOTests {

    @Test("Write and Read CSV")
    func testCSVWriteAndRead() async throws {
        let name = TypedColumn<String>(name: "name", values: ["Alice", "Bob, the Great", "Charlie"])
        let age  = TypedColumn<Int64>(name: "age", values: [25, 30, nil])
        let rate = TypedColumn<Double>(name: "rate", values: [1.2, nil, 3.4])
        let active = TypedColumn<Bool>(name: "active", values: [true, false, true])
        let df = try DataFrame(columns: [name, age, rate, active])

        let fm = FileManager.default
        let currentDir = URL(fileURLWithPath: fm.currentDirectoryPath)
        let fileURL = currentDir.appendingPathComponent("test_temp_df.csv")

        // Cleanup if exists
        if fm.fileExists(atPath: fileURL.path) {
            try fm.removeItem(at: fileURL)
        }

        defer {
            try? fm.removeItem(at: fileURL)
        }

        // Write
        try await df.writeCSV(to: fileURL)
        #expect(fm.fileExists(atPath: fileURL.path))

        // Read back
        var options = CSVReadOptions()
        options.inferTypes = true
        let readDf = try await DataFrame(csv: fileURL, options: options)

        #expect(readDf.shape.rows == 3)
        #expect(readDf.shape.columns == 4)
        #expect(readDf.columnNames == ["name", "age", "rate", "active"])

        // Check values
        let names = readDf[column: "name", as: String.self]?.values
        #expect(names == ["Alice", "Bob, the Great", "Charlie"])

        let ages = readDf[column: "age", as: Int64.self]?.values
        #expect(ages == [25, 30, nil])

        let rates = readDf[column: "rate", as: Double.self]?.values
        #expect(rates?[0] == 1.2)
        #expect(rates?[1] == nil)
        #expect(rates?[2] == 3.4)

        let actives = readDf[column: "active", as: Bool.self]?.values
        #expect(actives == [true, false, true])
    }

    @Test("Read JSON Array of Objects")
    func testJSONRead() async throws {
        let jsonContent = """
        [
            {"name": "Alice", "age": 25, "rate": 1.2, "active": true},
            {"name": "Bob", "age": 30, "active": false},
            {"name": "Charlie", "rate": 3.4, "active": true}
        ]
        """

        let fm = FileManager.default
        let currentDir = URL(fileURLWithPath: fm.currentDirectoryPath)
        let fileURL = currentDir.appendingPathComponent("test_temp_df.json")

        // Cleanup if exists
        if fm.fileExists(atPath: fileURL.path) {
            try fm.removeItem(at: fileURL)
        }

        defer {
            try? fm.removeItem(at: fileURL)
        }

        try jsonContent.write(to: fileURL, atomically: true, encoding: .utf8)

        // Read
        let df = try await DataFrame(json: fileURL)

        #expect(df.shape.rows == 3)
        #expect(df.shape.columns == 4)
        #expect(df.columnNames.contains("name"))
        #expect(df.columnNames.contains("age"))
        #expect(df.columnNames.contains("rate"))
        #expect(df.columnNames.contains("active"))

        let names = df[column: "name", as: String.self]?.values
        #expect(names == ["Alice", "Bob", "Charlie"])

        let ages = df[column: "age", as: Int64.self]?.values
        #expect(ages == [25, 30, nil])

        let rates = df[column: "rate", as: Double.self]?.values
        #expect(rates?[0] == 1.2)
        #expect(rates?[1] == nil)
        #expect(rates?[2] == 3.4)

        let actives = df[column: "active", as: Bool.self]?.values
        #expect(actives == [true, false, true])
    }

    @Test("CSV Reader handles custom null values")
    func testCSVNullValues() throws {
        let csvContent = """
        name,val
        Alice,10
        Bob,N/A
        Charlie,
        Dave,NaN
        """

        var options = CSVReadOptions()
        options.nullValues = ["", "N/A", "NaN"]
        options.inferTypes = true

        let df = try CSVReader.parse(csvContent, options: options)
        #expect(df.shape.rows == 4)
        
        let vals = df[column: "val", as: Int64.self]?.values
        #expect(vals == [10, nil, nil, nil])
    }

    @Test("CSV Reader/Writer handles custom delimiters, quotes, and newlines in quotes")
    func testCSVQuotesAndNewlinesAndCustomDelimiters() async throws {
        let csvContent = """
"name";"description";"value"
"Alice";"Hello\nWorld";10
"Bob";"This;is;fine";20
"Charlie";"Embedded ""quotes"" here";30
"""
        var readOptions = CSVReadOptions()
        readOptions.delimiter = ";"
        readOptions.inferTypes = true

        let df = try CSVReader.parse(csvContent, options: readOptions)
        #expect(df.shape.rows == 3)
        #expect(df.shape.columns == 3)

        let names = df[column: "name", as: String.self]?.values
        #expect(names == ["Alice", "Bob", "Charlie"])

        let descriptions = df[column: "description", as: String.self]?.values
        #expect(descriptions == ["Hello\nWorld", "This;is;fine", "Embedded \"quotes\" here"])

        let values = df[column: "value", as: Int64.self]?.values
        #expect(values == [10, 20, 30])

        // Let's write it back using a custom delimiter and read it again
        let fm = FileManager.default
        let currentDir = URL(fileURLWithPath: fm.currentDirectoryPath)
        let fileURL = currentDir.appendingPathComponent("test_temp_custom_df.csv")

        defer {
            try? fm.removeItem(at: fileURL)
        }

        try await CSVWriter.write(df, to: fileURL, delimiter: ";")
        
        let readBackDf = try await DataFrame(csv: fileURL, options: readOptions)
        #expect(readBackDf.shape.rows == 3)
        #expect(readBackDf.shape.columns == 3)
        #expect(readBackDf[column: "description", as: String.self]?.values == ["Hello\nWorld", "This;is;fine", "Embedded \"quotes\" here"])
    }
}
