import Foundation
import MLX
#if canImport(SwiftDataFrame)
import SwiftDataFrame
#endif

/// Parser for the SafeTensors file format.
/// Parses the JSON header and maps the binary data to MLXArrays.
public enum SafeTensorsParser {
    
    public static func parse(url: URL) throws -> [String: MLXArray] {
        let fileData = try Data(contentsOf: url, options: .mappedIfSafe)
        
        guard fileData.count >= 8 else {
            throw SwiftMLError.invalidInput("SafeTensors file too small (must be at least 8 bytes)")
        }
        
        // 1. Read little-endian 64-bit unsigned integer representing header length
        var headerLenVal: UInt64 = 0
        withUnsafeMutableBytes(of: &headerLenVal) { ptr in
            _ = fileData.copyBytes(to: ptr, from: 0..<8)
        }
        let headerLen = UInt64(littleEndian: headerLenVal)
        
        let headerStart = 8
        let headerEnd = 8 + Int(headerLen)
        
        guard fileData.count >= headerEnd else {
            throw SwiftMLError.invalidInput("SafeTensors header size exceeds file bounds")
        }
        
        // 2. Read and parse JSON header
        let headerData = fileData.subdata(in: headerStart..<headerEnd)
        guard let jsonDict = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw SwiftMLError.invalidInput("Failed to parse SafeTensors JSON header")
        }
        
        var tensors: [String: MLXArray] = [:]
        let binaryStart = headerEnd
        
        for (name, value) in jsonDict {
            if name == "__metadata__" { continue }
            
            guard let dict = value as? [String: Any],
                  let dtype = dict["dtype"] as? String,
                  let shape = dict["shape"] as? [Int],
                  let offsets = dict["data_offsets"] as? [Int],
                  offsets.count == 2 else {
                continue
            }
            
            let start = binaryStart + offsets[0]
            let end = binaryStart + offsets[1]
            
            guard end <= fileData.count else {
                throw SwiftMLError.invalidInput("Tensor '\(name)' offset out of bounds")
            }
            
            let tensorData = fileData.subdata(in: start..<end)
            
            let array: MLXArray
            switch dtype {
            case "F32":
                array = MLXArray(tensorData, shape, dtype: .float32)
            case "F16":
                array = MLXArray(tensorData, shape, dtype: .float16)
            case "I32":
                array = MLXArray(tensorData, shape, dtype: .int32)
            case "I64":
                array = MLXArray(tensorData, shape, dtype: .int64)
            default:
                array = MLXArray(tensorData, shape, dtype: .float32)
            }
            
            tensors[name] = array
        }
        
        return tensors
    }
}
