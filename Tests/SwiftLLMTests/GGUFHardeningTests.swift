#if os(macOS)
import Testing
import Foundation
@testable import SwiftLLM

@Suite("GGUF metadata and bounds hardening")
struct GGUFHardeningTests {
    private func le<T: FixedWidthInteger>(_ value: T) -> Data {
        var v = value.littleEndian
        return withUnsafeBytes(of: &v) { Data($0) }
    }
    private func string(_ text: String) -> Data { le(UInt64(text.utf8.count)) + Data(text.utf8) }

    private func write(_ data: Data) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gguf-\(UUID().uuidString).gguf")
        try data.write(to: url)
        return url
    }

    /// Header with `metadata` entries followed by one float32 tensor holding a single value.
    private func file(metadata: [Data], metadataCount: UInt64? = nil, dim: UInt64 = 1) -> Data {
        var data = Data([0x47, 0x47, 0x55, 0x46]) + le(UInt32(3)) + le(UInt64(1))
            + le(metadataCount ?? UInt64(metadata.count))
        metadata.forEach { data += $0 }
        data += string("t") + le(UInt32(1)) + le(dim) + le(UInt32(0)) + le(UInt64(0))
        while data.count % 32 != 0 { data.append(0) }
        data += le(Float(2.5).bitPattern)
        return data
    }

    @Test("Boolean metadata occupies one byte, so later entries stay aligned")
    func boolMetadataIsOneByte() throws {
        let flag = string("tokenizer.add_bos") + le(UInt32(7)) + Data([1])
        let name = string("general.name") + le(UInt32(8)) + string("tiny")
        let url = try write(file(metadata: [flag, name]))
        defer { try? FileManager.default.removeItem(at: url) }
        let tensors = try GGUFParser.parse(url: url)
        #expect(tensors["t"]?.asArray(Float.self) == [2.5])
    }

    @Test("GGML type 8 decodes Q8_0 and type 7 is not misread as Q8_0", arguments: [UInt32(8), UInt32(7)])
    func quantizedTypeIDs(type: UInt32) throws {
        var data = Data([0x47, 0x47, 0x55, 0x46]) + le(UInt32(3)) + le(UInt64(1)) + le(UInt64(0))
        data += string("output.weight") + le(UInt32(1)) + le(UInt64(32)) + le(type) + le(UInt64(0))
        while data.count % 32 != 0 { data.append(0) }
        data += le(Float16(0.5).bitPattern)
        data += Data((0..<32).map { UInt8(bitPattern: Int8($0 - 16)) })
        let url = try write(data)
        defer { try? FileManager.default.removeItem(at: url) }
        if type == 8 {
            let tensor = try #require(GGUFParser.parse(url: url)["output.weight"])
            #expect(tensor.scheme == .q8_0)
            #expect(tensor.asArray(Float.self) == (0..<32).map { Float($0 - 16) * 0.5 })
        } else {
            #expect(throws: (any Error).self) { try GGUFParser.parse(url: url) }
        }
    }

    @Test("Hostile array lengths, string lengths, and dimensions throw instead of hanging or trapping")
    func hostileInputs() throws {
        let hugeArray = string("a") + le(UInt32(9)) + le(UInt32(0)) + le(UInt64.max)
        let hugeString = string("b") + le(UInt32(8)) + le(UInt64.max)
        let truncatedScalar = string("c") + le(UInt32(10))   // UInt64 value cut off by EOF
        let inputs = [
            file(metadata: [hugeArray]),
            file(metadata: [hugeString]),
            file(metadata: [truncatedScalar], metadataCount: 40),
            file(metadata: [], dim: UInt64.max),
            file(metadata: [], dim: UInt64(Int32.max)),
        ]
        for data in inputs {
            let url = try write(data)
            defer { try? FileManager.default.removeItem(at: url) }
            #expect(throws: (any Error).self) { try GGUFParser.parse(url: url) }
        }
    }
}
#endif
