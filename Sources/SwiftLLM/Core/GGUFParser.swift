import Foundation
import MLX
#if canImport(SwiftDataFrame)
import SwiftDataFrame
#endif

/// Parser for the GGUF (GGML Unified Format) file format.
/// Decodes the metadata headers and maps the binary tensor blocks directly into MLXArrays.
public enum GGUFParser {
    
    public struct TensorInfo {
        public let name: String
        public let shape: [Int]
        public let type: UInt32
        public let offset: UInt64
    }
    
    private static func readInt<T: FixedWidthInteger>(_ type: T.Type, from data: Data, offset: inout Int) -> T {
        let size = MemoryLayout<T>.size
        let val = data.withUnsafeBytes { ptr -> T in
            ptr.loadUnaligned(fromByteOffset: offset, as: T.self)
        }
        offset += size
        return T(littleEndian: val)
    }
    
    public static func parse(url: URL) throws -> [String: MLXArray] {
        let fileData = try Data(contentsOf: url, options: .mappedIfSafe)
        
        guard fileData.count >= 24 else {
            throw SwiftMLError.invalidInput("GGUF file too small (must be at least 24 bytes)")
        }
        
        // 1. Read header magic: "GGUF" in ASCII
        let magic = fileData.subdata(in: 0..<4)
        guard magic == Data([0x47, 0x47, 0x55, 0x46]) else {
            throw SwiftMLError.invalidInput("Invalid GGUF magic number")
        }
        
        var offset = 4
        
        let version = readInt(UInt32.self, from: fileData, offset: &offset)
        guard version == 1 || version == 2 || version == 3 else {
            throw SwiftMLError.invalidInput("Unsupported GGUF version: \(version)")
        }
        
        let tensorCount = readInt(UInt64.self, from: fileData, offset: &offset)
        let metadataCount = readInt(UInt64.self, from: fileData, offset: &offset)
        
        func readString() throws -> String {
            guard offset + 8 <= fileData.count else {
                throw SwiftMLError.invalidInput("Unexpected EOF reading GGUF string length")
            }
            let len = readInt(UInt64.self, from: fileData, offset: &offset)
            
            let end = offset + Int(len)
            guard end <= fileData.count else {
                throw SwiftMLError.invalidInput("Unexpected EOF reading GGUF string content")
            }
            let strData = fileData.subdata(in: offset..<end)
            offset = end
            guard let str = String(data: strData, encoding: .utf8) else {
                throw SwiftMLError.invalidInput("Invalid UTF-8 in GGUF string")
            }
            return str
        }
        
        func skipValue(type: UInt32) throws {
            switch type {
            case 0...5: // UInt8, Int8, UInt16, Int16, UInt32, Int32
                let valSizes: [UInt32: Int] = [0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4]
                offset += valSizes[type] ?? 4
            case 6, 7: // Float32, Bool
                offset += 4
            case 8: // String
                _ = try readString()
            case 9: // Array
                guard offset + 12 <= fileData.count else {
                    throw SwiftMLError.invalidInput("Unexpected EOF reading GGUF array metadata")
                }
                let itemType = readInt(UInt32.self, from: fileData, offset: &offset)
                let len = readInt(UInt64.self, from: fileData, offset: &offset)
                for _ in 0..<len {
                    try skipValue(type: itemType)
                }
            case 10...13: // UInt64, Int64, Float64
                offset += 8
            default:
                throw SwiftMLError.invalidInput("Unknown GGUF metadata type: \(type)")
            }
        }
        
        // 2. Skip metadata pairs to reach tensor infos
        for _ in 0..<metadataCount {
            _ = try readString()
            guard offset + 4 <= fileData.count else {
                throw SwiftMLError.invalidInput("Unexpected EOF reading GGUF metadata type")
            }
            let valType = readInt(UInt32.self, from: fileData, offset: &offset)
            try skipValue(type: valType)
        }
        
        // 3. Parse tensor infos
        var tensorInfos: [TensorInfo] = []
        for _ in 0..<tensorCount {
            let name = try readString()
            
            guard offset + 4 <= fileData.count else {
                throw SwiftMLError.invalidInput("Unexpected EOF reading GGUF dimensions count")
            }
            let dimsCount = readInt(UInt32.self, from: fileData, offset: &offset)
            
            var dims: [Int] = []
            guard offset + Int(dimsCount) * 8 <= fileData.count else {
                throw SwiftMLError.invalidInput("Unexpected EOF reading GGUF dimensions")
            }
            for _ in 0..<dimsCount {
                let dim = readInt(UInt64.self, from: fileData, offset: &offset)
                dims.append(Int(dim))
            }
            
            guard offset + 12 <= fileData.count else {
                throw SwiftMLError.invalidInput("Unexpected EOF reading GGUF tensor type/offset")
            }
            let tensorType = readInt(UInt32.self, from: fileData, offset: &offset)
            let tensorOffset = readInt(UInt64.self, from: fileData, offset: &offset)
            
            tensorInfos.append(TensorInfo(name: name, shape: dims, type: tensorType, offset: tensorOffset))
        }
        
        // Align binary block start offset to 32 bytes boundary
        let alignment = 32
        let binaryStart = (offset + alignment - 1) & ~(alignment - 1)
        
        var tensors: [String: MLXArray] = [:]
        for info in tensorInfos {
            let start = binaryStart + Int(info.offset)
            
            let typeSize: Int
            let array: MLXArray
            let elementCount = info.shape.reduce(1, *)
            
            switch info.type {
            case 0: // Float32
                typeSize = 4
                let end = start + elementCount * typeSize
                guard end <= fileData.count else {
                    throw SwiftMLError.invalidInput("GGUF tensor '\(info.name)' offset out of bounds")
                }
                let tensorData = fileData.subdata(in: start..<end)
                array = MLXArray(tensorData, info.shape, dtype: .float32)
            case 1: // Float16
                typeSize = 2
                let end = start + elementCount * typeSize
                guard end <= fileData.count else {
                    throw SwiftMLError.invalidInput("GGUF tensor '\(info.name)' offset out of bounds")
                }
                let tensorData = fileData.subdata(in: start..<end)
                array = MLXArray(tensorData, info.shape, dtype: .float16)
            default:
                throw SwiftMLError.invalidInput("GGUF tensor '\(info.name)' uses unsupported quantization type \(info.type) (only unquantized Float32 and Float16 supported).")
            }
            
            tensors[info.name] = array
        }
        
        return tensors
    }
}
