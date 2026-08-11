import Testing
import Foundation
@testable import SwiftDataFrame

@Suite("SystemsCSVParser and High-Performance Columnar Tests")
struct SystemsCSVParserTests {

    @Test("SystemsCSVParser handles RFC 4180 quoted fields with embedded commas")
    func testQuotedCommas() {
        let csv = """
        1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
        2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
        """
        let data = csv.data(using: .utf8)!
        data.withUnsafeBytes { rawBuffer in
            let basePtr = rawBuffer.baseAddress!.assumingMemoryBound(to: UInt8.self)
            let buffer = UnsafeBufferPointer(start: basePtr, count: data.count)
            let parser = SystemsCSVParser()
            let records = parser.parse(buffer: buffer)

            #expect(records.count == 2)
            #expect(records[0].count == 12)
            #expect(records[1].count == 12)

            let name0 = VectorizedByteParsers.parseString(buffer: buffer, offset: records[0][3])
            #expect(name0 == "Braund, Mr. Owen Harris")

            let fare0 = VectorizedByteParsers.parseDouble(buffer: buffer, offset: records[0][9])
            #expect(fare0 == 7.25)

            let age0 = VectorizedByteParsers.parseInt(buffer: buffer, offset: records[0][5])
            #expect(age0 == 22)
        }
    }

    @Test("VectorizedByteParsers correctly decodes numeric types and strings")
    func testByteParsers() {
        let text = "-123.456, 789, \"Escaped \"\"Quote\"\" Text\""
        let data = text.data(using: .utf8)!
        data.withUnsafeBytes { rawBuffer in
            let basePtr = rawBuffer.baseAddress!.assumingMemoryBound(to: UInt8.self)
            let buffer = UnsafeBufferPointer(start: basePtr, count: data.count)
            
            let doubleOffset = CSVFieldOffset(startOffset: 0, length: 8, escapedQuotesPresent: false)
            let parsedDouble = VectorizedByteParsers.parseDouble(buffer: buffer, offset: doubleOffset)
            #expect(parsedDouble == -123.456)

            let intOffset = CSVFieldOffset(startOffset: 10, length: 3, escapedQuotesPresent: false)
            let parsedInt = VectorizedByteParsers.parseInt(buffer: buffer, offset: intOffset)
            #expect(parsedInt == 789)

            let strOffset = CSVFieldOffset(startOffset: 15, length: 24, escapedQuotesPresent: true)
            let parsedStr = VectorizedByteParsers.parseString(buffer: buffer, offset: strOffset)
            #expect(parsedStr == "Escaped \"Quote\" Text")
        }
    }

    @Test("TypedColumn Double vDSP reductions compute correct mean, variance, and stdDev")
    func testvDSPReductions() {
        let col = TypedColumn<Double>(name: "Fare", values: [10.0, 20.0, 30.0, 40.0, 50.0])
        #expect(col.mean() == 30.0)
        #expect(abs(col.variance() - 250.0) < 1e-9)
        #expect(abs(col.stdDev() - 15.8113883) < 1e-5)
    }

    @Test("DataFrame filterRows(by:) index-based filtering operates correctly")
    func testFilterByIndex() throws {
        let colA = TypedColumn<Double>(name: "val", values: [1.0, 2.0, 3.0, 4.0, 5.0])
        let df = try DataFrame(columns: [colA])
        let filtered = df.filterRows { index in index % 2 == 0 }

        #expect(filtered.shape.rows == 3)
        let vals: TypedColumn<Double>? = filtered[column: "val", as: Double.self]
        #expect(vals?.nonNullValues == [1.0, 3.0, 5.0])
    }

    @Test("SystemsCSVParser handles escaped quotes and missing trailing newline")
    func testEscapedQuotesAndMissingTrailingNewline() {
        let csv = "id,message,value\n1,\"Hello \"\"World\"\"\",100\n2,\"No newline at end\",200"
        let data = csv.data(using: .utf8)!
        data.withUnsafeBytes { rawBuffer in
            let basePtr = rawBuffer.baseAddress!.assumingMemoryBound(to: UInt8.self)
            let buffer = UnsafeBufferPointer(start: basePtr, count: data.count)
            let parser = SystemsCSVParser()
            let records = parser.parse(buffer: buffer)

            #expect(records.count == 3) // Header + 2 data rows
            #expect(records[1].count == 3)
            #expect(records[2].count == 3)

            let msg1 = VectorizedByteParsers.parseString(buffer: buffer, offset: records[1][1])
            #expect(msg1 == "Hello \"World\"")

            let val2 = VectorizedByteParsers.parseInt(buffer: buffer, offset: records[2][2])
            #expect(val2 == 200)
        }
    }
}

