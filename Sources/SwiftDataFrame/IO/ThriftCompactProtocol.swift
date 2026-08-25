import Foundation

/// Internal Thrift Compact Protocol binary decoder and encoder for Apache Parquet metadata.
internal enum ThriftCompact {

    enum TypeId: UInt8 {
        case stop = 0
        case booleanTrue = 1
        case booleanFalse = 2
        case byte = 3
        case i16 = 4
        case i32 = 5
        case i64 = 6
        case double = 7
        case binary = 8
        case list = 9
        case set = 10
        case map = 11
        case `struct` = 12
    }

    final class Reader {
        let bytes: [UInt8]
        var offset: Int = 0
        var lastFieldIdStack: [Int16] = [0]

        init(bytes: [UInt8]) {
            self.bytes = bytes
        }

        init(data: Data) {
            self.bytes = [UInt8](data)
        }

        var isEOF: Bool { offset >= bytes.count }

        func readByte() throws -> UInt8 {
            guard offset < bytes.count else {
                throw SwiftMLError.parseError(line: 0, description: "Unexpected EOF in Thrift byte stream")
            }
            let b = bytes[offset]
            offset += 1
            return b
        }

        func readVarint() throws -> UInt64 {
            var result: UInt64 = 0
            var shift: UInt64 = 0
            while !isEOF {
                let byte = try readByte()
                result |= UInt64(byte & 0x7F) << shift
                if (byte & 0x80) == 0 {
                    return result
                }
                shift += 7
                if shift > 63 {
                    throw SwiftMLError.parseError(line: 0, description: "Malformed Thrift varint")
                }
            }
            throw SwiftMLError.parseError(line: 0, description: "Unexpected EOF reading Thrift varint")
        }

        func readZigZagI32() throws -> Int32 {
            let u = try readVarint()
            let n = UInt32(truncatingIfNeeded: u)
            return Int32(bitPattern: (n >> 1) ^ (~(n & 1) &+ 1))
        }

        func readZigZagI64() throws -> Int64 {
            let u = try readVarint()
            return Int64(bitPattern: (u >> 1) ^ (~(u & 1) &+ 1))
        }

        func readDouble() throws -> Double {
            guard offset + 8 <= bytes.count else {
                throw SwiftMLError.parseError(line: 0, description: "Unexpected EOF reading Thrift double")
            }
            var raw: UInt64 = 0
            for i in 0..<8 {
                raw |= UInt64(bytes[offset + i]) << (i * 8)
            }
            offset += 8
            return Double(bitPattern: raw)
        }

        func readBinary() throws -> [UInt8] {
            let len = Int(try readVarint())
            guard offset + len <= bytes.count else {
                throw SwiftMLError.parseError(line: 0, description: "Unexpected EOF reading Thrift binary of length \(len)")
            }
            let sub = Array(bytes[offset..<(offset + len)])
            offset += len
            return sub
        }

        func readString() throws -> String {
            let bin = try readBinary()
            return String(decoding: bin, as: UTF8.self)
        }

        func readFieldBegin() throws -> (fieldId: Int16, type: TypeId) {
            let header = try readByte()
            if header == 0 {
                return (0, .stop)
            }

            let typeNibble = header & 0x0F
            guard let type = TypeId(rawValue: typeNibble) else {
                throw SwiftMLError.parseError(line: 0, description: "Unknown Thrift compact type: \(typeNibble)")
            }

            let delta = Int16((header >> 4) & 0x0F)
            let lastId = lastFieldIdStack.last ?? 0
            let fieldId: Int16
            if delta == 0 {
                fieldId = Int16(try readZigZagI32())
            } else {
                fieldId = lastId + delta
            }

            if !lastFieldIdStack.isEmpty {
                lastFieldIdStack[lastFieldIdStack.count - 1] = fieldId
            }
            return (fieldId, type)
        }

        func pushStruct() {
            lastFieldIdStack.append(0)
        }

        func popStruct() {
            _ = lastFieldIdStack.popLast()
        }

        func skip(type: TypeId) throws {
            switch type {
            case .stop:
                break
            case .booleanTrue, .booleanFalse:
                break
            case .byte:
                _ = try readByte()
            case .i16, .i32:
                _ = try readZigZagI32()
            case .i64:
                _ = try readZigZagI64()
            case .double:
                _ = try readDouble()
            case .binary:
                _ = try readBinary()
            case .list, .set:
                let header = try readByte()
                let elemTypeRaw = header & 0x0F
                guard let elemType = TypeId(rawValue: elemTypeRaw) else { return }
                var size = Int((header >> 4) & 0x0F)
                if size == 0x0F {
                    size = Int(try readVarint())
                }
                for _ in 0..<size {
                    try skip(type: elemType)
                }
            case .map:
                let size = Int(try readVarint())
                if size > 0 {
                    let header = try readByte()
                    let kType = TypeId(rawValue: (header >> 4) & 0x0F) ?? .stop
                    let vType = TypeId(rawValue: header & 0x0F) ?? .stop
                    for _ in 0..<size {
                        try skip(type: kType)
                        try skip(type: vType)
                    }
                }
            case .struct:
                pushStruct()
                while true {
                    let field = try readFieldBegin()
                    if field.type == .stop { break }
                    try skip(type: field.type)
                }
                popStruct()
            }
        }
    }

    final class Writer {
        var bytes: [UInt8] = []
        var lastFieldIdStack: [Int16] = [0]

        func writeByte(_ b: UInt8) {
            bytes.append(b)
        }

        func writeVarint(_ val: UInt64) {
            var v = val
            while v >= 0x80 {
                bytes.append(UInt8((v & 0x7F) | 0x80))
                v >>= 7
            }
            bytes.append(UInt8(v & 0x7F))
        }

        func writeZigZagI32(_ n: Int32) {
            let u = UInt32(bitPattern: (n << 1) ^ (n >> 31))
            writeVarint(UInt64(u))
        }

        func writeZigZagI64(_ n: Int64) {
            let u = UInt64(bitPattern: (n << 1) ^ (n >> 63))
            writeVarint(u)
        }

        func writeBinary(_ bin: [UInt8]) {
            writeVarint(UInt64(bin.count))
            bytes.append(contentsOf: bin)
        }

        func writeString(_ str: String) {
            writeBinary([UInt8](str.utf8))
        }

        func pushStruct() {
            lastFieldIdStack.append(0)
        }

        func popStruct() {
            _ = lastFieldIdStack.popLast()
        }

        func writeFieldBegin(fieldId: Int16, type: TypeId) {
            let lastId = lastFieldIdStack.last ?? 0
            let delta = fieldId - lastId
            if delta > 0 && delta <= 15 {
                writeByte(UInt8((delta << 4) | Int16(type.rawValue)))
            } else {
                writeByte(type.rawValue)
                writeZigZagI32(Int32(fieldId))
            }
            if !lastFieldIdStack.isEmpty {
                lastFieldIdStack[lastFieldIdStack.count - 1] = fieldId
            }
        }

        func writeFieldStop() {
            writeByte(0)
        }

        func writeListBegin(elemType: TypeId, size: Int) {
            if size < 15 {
                writeByte(UInt8((size << 4) | Int(elemType.rawValue)))
            } else {
                writeByte(UInt8(0xF0 | Int(elemType.rawValue)))
                writeVarint(UInt64(size))
            }
        }
    }
}
