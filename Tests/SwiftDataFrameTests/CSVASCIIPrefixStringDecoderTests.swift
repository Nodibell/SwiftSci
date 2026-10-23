import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV bounded ASCII String null keys")
struct CSVASCIIPrefixStringDecoderTests {
    @Test("Preparation uses the selected row budget and rejects unsuitable sets")
    func eligibility() {
        let tokens: Set<String> = ["", "NA", "null"]
        #expect(CSVASCIIPrefixStringDecoder(nullValues: tokens, selectedRowCount: 11) == nil)
        #expect(CSVASCIIPrefixStringDecoder(nullValues: tokens, selectedRowCount: 12) != nil)
        #expect(CSVASCIIPrefixStringDecoder(nullValues: tokens, selectedRowCount: 13) != nil)
        let rejected: [Set<String>] = [[], ["NA"], ["", "NA"], ["", "NA", "é"]]
        for values in rejected {
            #expect(CSVASCIIPrefixStringDecoder(nullValues: values, selectedRowCount: 1_000) == nil)
        }
        let units = [unichar](repeating: 0x0100, count: 128)
        let cocoa = units.withUnsafeBufferPointer { pointer in
            NSString(characters: pointer.baseAddress!, length: pointer.count)
        }
        let bridged = cocoa as String
        let bridgedTokens: Set<String> = ["", "NA", bridged]
        #expect(CSVASCIIPrefixStringDecoder(nullValues: bridgedTokens, selectedRowCount: 1_000) == nil)
        let long: Set<String> = ["", "NA", String(repeating: "x", count: 10_000)]
        #expect(CSVASCIIPrefixStringDecoder(nullValues: long, selectedRowCount: 64) == nil)
        var saturated = Set<String>()
        for byte in UInt8(0)...UInt8(127) { saturated.insert(String(UnicodeScalar(byte))) }
        #expect(CSVASCIIPrefixStringDecoder(nullValues: saturated, selectedRowCount: 384) == nil)
    }

    @Test("Prepared keys own values and preserve parsed String membership")
    func ownedMembership() throws {
        let expected: Set<String> = ["", "K", "NA", "?", "@", "\u{7f}"]
        var tokens = expected
        let decoder = try #require(CSVASCIIPrefixStringDecoder(nullValues: tokens, selectedRowCount: 128))
        tokens.removeAll()
        tokens.insert("replacement")
        let strings: [String] = ["", " \tNA\r\n", "na", "é", "e\u{301}", "K", "K", "?", "@", "\u{7f}", "\0", "\"a\"\"b\"", "汉字"]
        var inputs: [[UInt8]] = strings.map { (value: String) -> [UInt8] in Array(value.utf8) }
        inputs.append([0xff])
        inputs.append([34, 0xff, 34])
        inputs.append([34, 0xff, 34, 34, 34])
        for bytes in inputs {
            bytes.withUnsafeBufferPointer { buffer in
                let field = CSVFieldOffset(startOffset: 0, length: bytes.count, escapedQuotesPresent: bytes.contains(34))
                let text = VectorizedByteParsers.parseString(buffer: buffer, offset: field)
                switch decoder.decode(buffer: buffer, offset: field) {
                case .missing: #expect(expected.contains(text))
                case .value(let value):
                    #expect(!expected.contains(text))
                    #expect(value == text)
                case .invalid: Issue.record("String conversion unexpectedly rejected a field")
                }
            }
        }
    }
}
