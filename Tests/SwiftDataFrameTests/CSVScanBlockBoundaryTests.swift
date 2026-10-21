import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV scan block boundaries")
struct CSVScanBlockBoundaryTests {
    @Test(arguments: 0..<32, [UInt8(44), 59, 13, 0, 255])
    func structuralOffsets(padding: Int, delimiter: UInt8) {
        for length in 0..<32 {
            var bytes = [UInt8](repeating: 65, count: padding)
            for row in 0..<32 {
                bytes += [UInt8](repeating: 66, count: length)
                bytes += [delimiter, 13, 10]
                if row % 3 == 0 { bytes += [delimiter, delimiter, 10] }
            }
            bytes.withUnsafeBufferPointer { input in
                let parser = SystemsCSVParser(delimiterByte: delimiter)
                let expected = parser.parseIndex(buffer: input, minimumParallelBytes: .max)
                let actual = parser.parseIndex(buffer: input, minimumParallelBytes: 0)
                #expect(actual.rowStarts == expected.rowStarts)
                #expect(actual.testFields.map(\.startOffset) == expected.testFields.map(\.startOffset))
                #expect(actual.testFields.map(\.length) == expected.testFields.map(\.length))
            }
        }
    }

    @Test(arguments: 0..<32)
    func lateQuotesDiscardAllChunks(offset: Int) {
        for quote: UInt8 in [34, 39, 0, 255] {
            var bytes = Array(String(repeating: "123,456\r\n", count: 128).utf8)
            bytes += [UInt8](repeating: 65, count: offset) + [quote] + Array("x,y\nz".utf8) + [quote, 10]
            bytes.withUnsafeBufferPointer { input in
                let parser = SystemsCSVParser(quoteByte: quote)
                #expect(parser.parseParallelIndex(buffer: input) == nil)
                let expected = parser.parseIndex(buffer: input, minimumParallelBytes: .max)
                let actual = parser.parseIndex(buffer: input, minimumParallelBytes: 0)
                #expect(actual.rowStarts == expected.rowStarts)
                #expect(actual.testFields.map(\.startOffset) == expected.testFields.map(\.startOffset))
                #expect(actual.testFields.map(\.length) == expected.testFields.map(\.length))
                #expect(actual.testFields.map(\.escapedQuotesPresent) == expected.testFields.map(\.escapedQuotesPresent))
            }
        }
    }

    @Test(arguments: [false, true])
    func automaticCutover(quoted: Bool) {
        let threshold = 1024 * 1024
        let template = Array(String(repeating: "a,b,c\r\n", count: threshold / 7 + 16).utf8)
        for length in [threshold - 1, threshold, threshold + 1] {
            var bytes = Array(template.prefix(length))
            if quoted { bytes[length - 3] = 34 }
            bytes.withUnsafeBufferPointer { input in
                let parser = SystemsCSVParser()
                let expected = parser.parseIndex(buffer: input, minimumParallelBytes: .max)
                let actual = parser.parseIndex(buffer: input)
                #expect(actual.rowStarts == expected.rowStarts)
                #expect(actual.testFields.map(\.startOffset) == expected.testFields.map(\.startOffset))
                #expect(actual.testFields.map(\.length) == expected.testFields.map(\.length))
                #expect(actual.testFields.map(\.escapedQuotesPresent) == expected.testFields.map(\.escapedQuotesPresent))
            }
        }
    }


    @Test func unevenRecordLengths() {
        for tail in ["", "\n", "last,field\n"] {
            let text = "a,b\n" + String(repeating: "x", count: 20_000) + ",1\n" + String(repeating: "y,2\n", count: 100) + tail
            Array(text.utf8).withUnsafeBufferPointer { input in
                let parser = SystemsCSVParser()
                let expected = parser.parseIndex(buffer: input, minimumParallelBytes: .max)
                let actual = parser.parseIndex(buffer: input, minimumParallelBytes: 0)
                #expect(actual.rowStarts == expected.rowStarts)
                #expect(actual.testFields.map(\.startOffset) == expected.testFields.map(\.startOffset))
                #expect(actual.testFields.map(\.length) == expected.testFields.map(\.length))
            }
        }
    }

}
