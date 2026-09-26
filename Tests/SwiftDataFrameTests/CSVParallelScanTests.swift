import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("Parallel CSV scan")
struct CSVParallelScanTests {
    @Test("Chunked and serial scanning have identical byte coordinates")
    func parity() {
        for ending in ["\n", "\r\n"] {
            for tail in ["last,1,2", "last,1,", "last,1,2\n", ""] {
                let body = (0..<400).map { row in
                    row % 7 == 0 ? "\(row),," : row % 11 == 0 ? "\(row)" : "\(row),\(row * 2),x,y"
                }.joined(separator: ending)
                let data = Data((body + ending + tail).utf8)
                data.withUnsafeBytes { raw in
                    let buffer = raw.bindMemory(to: UInt8.self)
                    let parser = SystemsCSVParser()
                    let serial = parser.parseIndex(buffer: buffer, minimumParallelBytes: .max)
                    let parallel = parser.parseIndex(buffer: buffer, minimumParallelBytes: 0)
                    #expect(serial.rowStarts == parallel.rowStarts)
                    #expect(serial.fields.map(\.startOffset) == parallel.fields.map(\.startOffset))
                    #expect(serial.fields.map(\.length) == parallel.fields.map(\.length))
                    #expect(serial.fields.map(\.escapedQuotesPresent) == parallel.fields.map(\.escapedQuotesPresent))
                }
            }
        }
    }

    @Test("Quotes force the full DFA even across candidate chunk boundaries")
    func quotedFallback() {
        for quoted in ["\"a,b\"", "\"line\nline\"", "\"escaped \"\" quote\"", "\"unfinished\n"] {
            let text = String(repeating: "a,b,c\n", count: 100) + quoted + ",d,e\n" + String(repeating: "x,y,z\n", count: 100)
            Data(text.utf8).withUnsafeBytes { raw in
                let buffer = raw.bindMemory(to: UInt8.self)
                let parser = SystemsCSVParser()
                #expect(parser.parseParallelIndex(buffer: buffer) == nil)
                let serial = parser.parseIndex(buffer: buffer, minimumParallelBytes: .max)
                let fallback = parser.parseIndex(buffer: buffer, minimumParallelBytes: 0)
                #expect(serial.rowStarts == fallback.rowStarts)
                #expect(serial.fields.map(\.startOffset) == fallback.fields.map(\.startOffset))
                #expect(serial.fields.map(\.length) == fallback.fields.map(\.length))
            }
        }
    }

    @Test("Custom delimiter, empty input, single long record and blank records")
    func boundaries() {
        for text in ["", "\n", String(repeating: "\n", count: 200), String(repeating: "x", count: 10000), String(repeating: "a;b;\n", count: 200)] {
            Data(text.utf8).withUnsafeBytes { raw in
                let buffer = raw.bindMemory(to: UInt8.self)
                let parser = SystemsCSVParser(delimiterByte: 59)
                let serial = parser.parseIndex(buffer: buffer, minimumParallelBytes: .max)
                let parallel = parser.parseIndex(buffer: buffer, minimumParallelBytes: 0)
                #expect(serial.rowStarts == parallel.rowStarts)
                #expect(serial.fields.map(\.startOffset) == parallel.fields.map(\.startOffset))
                #expect(serial.fields.map(\.length) == parallel.fields.map(\.length))
            }
        }
    }
    @Test("Unusual delimiters and random bytes retain serial semantics")
    func randomizedParity() {
        var state: UInt64 = 0x89ABCDEF
        for delimiter: UInt8 in [10, 13, 44, 59] {
            for iteration in 0..<100 {
                var bytes = [UInt8]()
                let alphabet: [UInt8] = [0, 9, 10, 13, 32, 44, 59, 65, 127, 195, 169]
                for _ in 0..<(100 + iteration * 11) {
                    state = state &* 6364136223846793005 &+ 1
                    bytes.append(alphabet[Int(state >> 32) % alphabet.count])
                }
                bytes.withUnsafeBufferPointer { buffer in
                    let parser = SystemsCSVParser(delimiterByte: delimiter)
                    let serial = parser.parseIndex(buffer: buffer, minimumParallelBytes: .max)
                    let parallel = parser.parseIndex(buffer: buffer, minimumParallelBytes: 0)
                    #expect(serial.rowStarts == parallel.rowStarts)
                    #expect(serial.fields.map(\.startOffset) == parallel.fields.map(\.startOffset))
                    #expect(serial.fields.map(\.length) == parallel.fields.map(\.length))
                }
            }
        }
    }

}
