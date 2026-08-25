import Foundation

/// A pure-Swift zero-dependency Snappy decompressor and compressor.
///
/// Implements Google Snappy raw block compression and decompression format according to the Snappy specification.
public enum SnappyDecompressor: Sendable {

    /// Decompresses raw Snappy-compressed bytes.
    ///
    /// - Parameter data: Snappy-compressed data block.
    /// - Returns: Decompressed uncompressed `Data`.
    /// - Throws: `SwiftMLError.parseError` if the byte stream is malformed or corrupted.
    public static func decompress(data: Data) throws -> Data {
        guard !data.isEmpty else { return Data() }

        return try data.withUnsafeBytes { rawBuffer in
            guard let ptr = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return Data()
            }
            let count = rawBuffer.count
            var offset = 0

            // 1. Read varint32 uncompressed length
            var uncompressedLen = 0
            var shift = 0
            while offset < count {
                let byte = ptr[offset]
                offset += 1
                uncompressedLen |= Int(byte & 0x7F) << shift
                if (byte & 0x80) == 0 {
                    break
                }
                shift += 7
                if shift > 35 {
                    throw SwiftMLError.parseError(line: 0, description: "Corrupted Snappy varint header")
                }
            }

            var output = [UInt8]()
            output.reserveCapacity(uncompressedLen)

            // 2. Decode elements (literals and copies)
            while offset < count {
                let tagByte = ptr[offset]
                offset += 1
                let elementTag = tagByte & 0x03

                switch elementTag {
                case 0x00:
                    // Literal
                    var literalLen = Int(tagByte >> 2) + 1
                    if literalLen > 60 {
                        let extraBytesCount = literalLen - 60
                        guard offset + extraBytesCount <= count else {
                            throw SwiftMLError.parseError(line: 0, description: "Unexpected EOF reading Snappy literal length")
                        }
                        literalLen = 0
                        for b in 0..<extraBytesCount {
                            literalLen |= Int(ptr[offset + b]) << (b * 8)
                        }
                        literalLen += 1
                        offset += extraBytesCount
                    }

                    guard offset + literalLen <= count else {
                        throw SwiftMLError.parseError(line: 0, description: "Snappy literal exceeds buffer bound")
                    }
                    output.append(contentsOf: UnsafeBufferPointer(start: ptr + offset, count: literalLen))
                    offset += literalLen

                case 0x01:
                    // Copy with 1-byte offset
                    let length = Int((tagByte >> 2) & 0x07) + 4
                    guard offset < count else {
                        throw SwiftMLError.parseError(line: 0, description: "Unexpected EOF reading Snappy copy offset")
                    }
                    let offsetByte = ptr[offset]
                    offset += 1
                    let copyOffset = (Int(tagByte & 0xE0) << 3) | Int(offsetByte)
                    guard copyOffset > 0 && copyOffset <= output.count else {
                        throw SwiftMLError.parseError(line: 0, description: "Invalid Snappy copy offset \(copyOffset)")
                    }

                    let startIdx = output.count - copyOffset
                    for i in 0..<length {
                        output.append(output[startIdx + (i % copyOffset)])
                    }

                case 0x02:
                    // Copy with 2-byte offset
                    let length = Int(tagByte >> 2) + 1
                    guard offset + 2 <= count else {
                        throw SwiftMLError.parseError(line: 0, description: "Unexpected EOF reading Snappy 2-byte copy offset")
                    }
                    let copyOffset = Int(ptr[offset]) | (Int(ptr[offset + 1]) << 8)
                    offset += 2
                    guard copyOffset > 0 && copyOffset <= output.count else {
                        throw SwiftMLError.parseError(line: 0, description: "Invalid Snappy copy offset \(copyOffset)")
                    }

                    let startIdx = output.count - copyOffset
                    for i in 0..<length {
                        output.append(output[startIdx + (i % copyOffset)])
                    }

                case 0x03:
                    // Copy with 4-byte offset
                    let length = Int(tagByte >> 2) + 1
                    guard offset + 4 <= count else {
                        throw SwiftMLError.parseError(line: 0, description: "Unexpected EOF reading Snappy 4-byte copy offset")
                    }
                    let copyOffset = Int(ptr[offset]) | (Int(ptr[offset + 1]) << 8) | (Int(ptr[offset + 2]) << 16) | (Int(ptr[offset + 3]) << 24)
                    offset += 4
                    guard copyOffset > 0 && copyOffset <= output.count else {
                        throw SwiftMLError.parseError(line: 0, description: "Invalid Snappy copy offset \(copyOffset)")
                    }

                    let startIdx = output.count - copyOffset
                    for i in 0..<length {
                        output.append(output[startIdx + (i % copyOffset)])
                    }

                default:
                    break
                }
            }

            return Data(output)
        }
    }

    /// Compresses raw bytes using Snappy LZ77 format.
    ///
    /// - Parameter data: Uncompressed data.
    /// - Returns: Snappy-compressed `Data`.
    public static func compress(data: Data) -> Data {
        guard !data.isEmpty else { return Data([0x00]) }

        return data.withUnsafeBytes { rawBuffer in
            guard let ptr = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return Data([0x00])
            }
            let count = rawBuffer.count

            var output = [UInt8]()
            // 1. Write varint32 uncompressed length
            var len = count
            while len >= 0x80 {
                output.append(UInt8((len & 0x7F) | 0x80))
                len >>= 7
            }
            output.append(UInt8(len & 0x7F))

            // 2. LZ77 Compression Table
            let tableSize = 4096
            var table = [Int](repeating: -1, count: tableSize)

            func emitLiteral(start: Int, end: Int) {
                let litLen = end - start
                guard litLen > 0 else { return }
                let tagLen = litLen - 1
                if tagLen < 60 {
                    output.append(UInt8(tagLen << 2))
                } else if tagLen <= 0xFF {
                    output.append(UInt8(60 << 2))
                    output.append(UInt8(tagLen & 0xFF))
                } else {
                    output.append(UInt8(61 << 2))
                    output.append(UInt8(tagLen & 0xFF))
                    output.append(UInt8((tagLen >> 8) & 0xFF))
                }
                output.append(contentsOf: UnsafeBufferPointer(start: ptr + start, count: litLen))
            }

            var litStart = 0
            var i = 0

            while i + 4 <= count {
                let h = ((Int(ptr[i]) | (Int(ptr[i + 1]) << 8) | (Int(ptr[i + 2]) << 16) | (Int(ptr[i + 3]) << 24)) &* 0x1e35a7bd) & (tableSize - 1)
                let matchPos = table[h]
                table[h] = i

                if matchPos >= 0 && (i - matchPos) < 65535 && (i - matchPos) > 0 {
                    // Check if 4 bytes match
                    if ptr[matchPos] == ptr[i] &&
                       ptr[matchPos + 1] == ptr[i + 1] &&
                       ptr[matchPos + 2] == ptr[i + 2] &&
                       ptr[matchPos + 3] == ptr[i + 3] {

                        // Measure match length
                        var matchLen = 4
                        while (i + matchLen < count) && (ptr[matchPos + matchLen] == ptr[i + matchLen]) && (matchLen < 64) {
                            matchLen += 1
                        }

                        // Emit preceding literals
                        if i > litStart {
                            emitLiteral(start: litStart, end: i)
                        }

                        let copyOffset = i - matchPos
                        if matchLen >= 4 && matchLen <= 11 && copyOffset <= 2047 {
                            // Element 0x01: length (3 bits = length - 4), offset (11 bits)
                            let tagByte = UInt8(0x01 | ((matchLen - 4) << 2) | ((copyOffset >> 8) << 5))
                            output.append(tagByte)
                            output.append(UInt8(copyOffset & 0xFF))
                        } else {
                            // Element 0x02: length (6 bits = length - 1), offset (16 bits)
                            let tagByte = UInt8(0x02 | ((matchLen - 1) << 2))
                            output.append(tagByte)
                            output.append(UInt8(copyOffset & 0xFF))
                            output.append(UInt8((copyOffset >> 8) & 0xFF))
                        }

                        i += matchLen
                        litStart = i
                        continue
                    }
                }
                i += 1
            }

            // Emit remaining trailing literals
            if litStart < count {
                emitLiteral(start: litStart, end: count)
            }

            return Data(output)
        }
    }
}
