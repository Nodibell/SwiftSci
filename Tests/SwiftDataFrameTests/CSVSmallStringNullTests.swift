import Foundation
import Testing
@testable import SwiftDataFrame

@Suite("CSV inline String null keys")
struct CSVSmallStringNullTests {
    @Test("Inline keys accept only one or two distinct tokens")
    func eligibility() {
        #expect(CSVSmallStringDecoder(nullValues: []) == nil)
        #expect(CSVSmallStringDecoder(nullValues: ["NA"]) != nil)
        #expect(CSVSmallStringDecoder(nullValues: ["", "NA"]) != nil)
        #expect(CSVSmallStringDecoder(nullValues: ["", "NA", "null"]) == nil)
    }

    @Test("Inline keys own their values and retain String equality")
    func ownedMembership() throws {
        let tokenSets: [Set<String>] = [[""], ["NA"], ["", "NA"], ["é", "K"], ["a\"b", "�"]]
        let strings: [String] = ["", " \tNA\r\n", "na", "é", "e\u{301}", "K", "K", "\"a\"\"b\"", "汉字"]
        var inputs: [[UInt8]] = strings.map { (value: String) -> [UInt8] in Array(value.utf8) }
        inputs.append([0xff])
        inputs.append([34, 0xff, 34])
        for expectedTokens in tokenSets {
            var tokens = expectedTokens
            let decoder = try #require(CSVSmallStringDecoder(nullValues: tokens))
            tokens.removeAll()
            tokens.insert("replacement")
            for bytes in inputs {
                bytes.withUnsafeBufferPointer { buffer in
                    let field = CSVFieldOffset(startOffset: 0, length: bytes.count, escapedQuotesPresent: bytes.contains(34))
                    let text = VectorizedByteParsers.parseString(buffer: buffer, offset: field)
                    switch decoder.decode(buffer: buffer, offset: field) {
                    case .missing: #expect(expectedTokens.contains(text))
                    case .value(let value):
                        #expect(!expectedTokens.contains(text))
                        #expect(value == text)
                    case .invalid: Issue.record("String conversion unexpectedly rejected a field")
                    }
                }
            }
        }
    }
}
