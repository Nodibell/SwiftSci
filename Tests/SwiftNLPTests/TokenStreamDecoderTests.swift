import Testing
@testable import SwiftNLP

@Suite("Incremental UTF-8 decoding")
struct TokenStreamDecoderTests {
    private func streamed(_ chunks: [[UInt8]]) -> String {
        var decoder = TokenStreamDecoder(decodeBytes: { chunks[$0] })
        var result = ""
        for index in chunks.indices { result += decoder.append(index) ?? "" }
        result += decoder.finish()
        #expect(decoder.finish().isEmpty)
        return result
    }

    @Test("Every two-byte input matches standard replacement decoding")
    func allPairs() {
        for first in UInt16(0)...255 {
            for second in UInt16(0)...255 {
                let bytes = [UInt8(first), UInt8(second)]
                let actual = streamed(bytes.map { [$0] })
                #expect(Array(actual.utf8) == Array(String(decoding: bytes, as: UTF8.self).utf8))
            }
        }
    }

    @Test("All split points preserve valid scalars and repair invalid or unfinished sequences")
    func splitPoints() {
        let cases: [[UInt8]] = [
            Array(" café e\u{301} 世界 👩‍🔬 🇺🇦 �\r\n".utf8),
            [0xe0, 0xa0, 0x80], [0xed, 0x9f, 0xbf], [0xf0, 0x90, 0x80, 0x80],
            [0xf4, 0x8f, 0xbf, 0xbf], [0xe0, 0x9f, 0x80], [0xed, 0xa0, 0x80],
            [0xf0, 0x8f, 0xbf, 0xbf], [0xf4, 0x90, 0x80, 0x80], [0xff, 0x80],
            [0x61, 0xe4, 0xb8], [0xf0, 0x90, 0x80], [0xe4, 0x62, 0x80],
            [0xe4, 0xb8, 0xc2, 0xa2], [0xf0, 0x90, 0x80, 0x61]
        ]
        for bytes in cases {
            for split in 0...bytes.count {
                let actual = streamed([Array(bytes[..<split]), Array(bytes[split...])])
                #expect(Array(actual.utf8) == Array(String(decoding: bytes, as: UTF8.self).utf8))
            }
        }
    }

    @Test("Incomplete scalars wait; complete prefixes are emitted immediately")
    func incrementalOutput() {
        let chunks: [[UInt8]] = [[0x41, 0xe4], [0xb8], [0x96], [0xef], [0xbf], [0xbd]]
        var decoder = TokenStreamDecoder(decodeBytes: { chunks[$0] })
        #expect(decoder.append(0) == "A")
        #expect(decoder.append(1) == nil)
        #expect(decoder.append(2) == "世")
        #expect(decoder.append(3) == nil)
        #expect(decoder.append(4) == nil)
        #expect(decoder.append(5) == "�")
        #expect(decoder.finish().isEmpty)
    }
    @Test("An empty token remains observable while UTF-8 bytes are pending")
    func emptyTokenWithPendingBytes() {
        var decoder = TokenStreamDecoder(decodeBytes: { $0 == 1 ? [0xe4] : [] })
        #expect(decoder.append(1) == nil)
        #expect(decoder.append(2) == "")
        #expect(decoder.finish() == "�")
        #expect(decoder.finish().isEmpty)
    }

}
