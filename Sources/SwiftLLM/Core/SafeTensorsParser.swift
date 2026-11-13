#if os(macOS)
import Foundation
import MLX
#if canImport(SwiftDataFrame)
import SwiftDataFrame
#endif

/// Parser for the SafeTensors file format.
/// Parses the JSON header and maps the binary data to MLXArrays.
public enum SafeTensorsParser {
    
    /// Parse.
    /// - Parameters:
    ///   - url: The url.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[String: MLXArray]` result.
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
        // Reject hostile lengths before converting to Int (the conversion or the
        // addition would otherwise trap on 64-bit overflow).
        guard headerLen <= UInt64(fileData.count - headerStart) else {
            throw SwiftMLError.invalidInput("SafeTensors header size exceeds file bounds")
        }
        let headerEnd = headerStart + Int(headerLen)
        
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
            
            guard offsets[0] >= 0, offsets[0] <= offsets[1],
                  offsets[1] <= fileData.count - binaryStart else {
                throw SwiftMLError.invalidInput("Tensor '\(name)' offset out of bounds")
            }
            let start = binaryStart + offsets[0]
            let end = binaryStart + offsets[1]
            
            guard end <= fileData.count else {
                throw SwiftMLError.invalidInput("Tensor '\(name)' offset out of bounds")
            }
            
            let tensorData = fileData.subdata(in: start..<end)

            let mlxDType: DType
            switch dtype {
            case "F32": mlxDType = .float32
            case "F16": mlxDType = .float16
            case "I32": mlxDType = .int32
            case "I64": mlxDType = .int64
            case "BF16": mlxDType = .bfloat16
            case "I8": mlxDType = .int8
            // Unsigned types appear in MLX-quantized checkpoints, where `U32`
            // stores bit-packed weights alongside `scales` / `biases` tensors.
            case "U8": mlxDType = .uint8
            case "U16": mlxDType = .uint16
            case "U32": mlxDType = .uint32
            case "U64": mlxDType = .uint64
            case "I16": mlxDType = .int16
            case "BOOL": mlxDType = .bool
            default:
                // Strict validation: unknown dtypes must NOT be silently coerced
                // to float32 — this would cause silent data corruption.
                throw SwiftMLError.invalidInput(
                    "SafeTensors: unsupported dtype '\(dtype)' for tensor '\(name)'. " +
                    "Supported: F32, F16, BF16, I8, I16, I32, I64, U8, U16, U32, U64, BOOL."
                )
            }

            // MLXArray(data, shape, dtype:) traps in shapePrecondition on negative
            // dims, integer overflow, or mismatched byte lengths; validate up front.
            var elementCount = 1
            for dim in shape {
                guard dim >= 0 else {
                    throw SwiftMLError.invalidInput(
                        "SafeTensors: negative dimension \(dim) in shape \(shape) for tensor '\(name)'."
                    )
                }
                let (product, overflow) = elementCount.multipliedReportingOverflow(by: dim)
                guard !overflow else {
                    throw SwiftMLError.invalidInput(
                        "SafeTensors: shape \(shape) overflows element count for tensor '\(name)'."
                    )
                }
                elementCount = product
            }
            let (expectedBytes, byteOverflow) = elementCount.multipliedReportingOverflow(by: mlxDType.size)
            guard !byteOverflow, expectedBytes == tensorData.count else {
                throw SwiftMLError.invalidInput(
                    "SafeTensors: byte count mismatch for tensor '\(name)' (\(dtype) \(shape)): " +
                    "expected \(byteOverflow ? "overflow" : "\(expectedBytes)") bytes, got \(tensorData.count)."
                )
            }

            tensors[name] = MLXArray(tensorData, shape, dtype: mlxDType)
        }
        
        return tensors
    }
}
#endif // os(macOS)
