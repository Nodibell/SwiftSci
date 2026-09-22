#if os(macOS)
import Foundation
import MLX
#if canImport(SwiftDataFrame)
import SwiftDataFrame
#endif

/// Scheme-specific layout metadata for quantized tensors.
public struct QuantizedLayoutMetadata: Sendable, Equatable {
    /// Number of weights represented by each quantized block (e.g. 32 for Q4_0, Q8_0).
    public var blockSize: Int
    /// Total byte size of each block in raw storage (e.g. 18 for Q4_0, 34 for Q8_0).
    public var bytesPerBlock: Int
    /// Format of scales within the block layout.
    public var scaleLayout: ScaleLayout

    public enum ScaleLayout: Sendable, Equatable {
        /// 2-byte FP16 scale prefixed before nibbles/bytes in each block (Q4_0, Q8_0).
        case embeddedFp16Prefix
        /// 2-byte FP16 scale and 2-byte FP16 min prefixed before nibbles (Q4_1).
        case embeddedFp16WithMin
        /// Unquantized float representation.
        case none
    }

    public init(blockSize: Int, bytesPerBlock: Int, scaleLayout: ScaleLayout) {
        self.blockSize = blockSize
        self.bytesPerBlock = bytesPerBlock
        self.scaleLayout = scaleLayout
    }

    public static let q4_0 = QuantizedLayoutMetadata(blockSize: 32, bytesPerBlock: 18, scaleLayout: .embeddedFp16Prefix)
    public static let q4_1 = QuantizedLayoutMetadata(blockSize: 32, bytesPerBlock: 20, scaleLayout: .embeddedFp16WithMin)
    public static let q8_0 = QuantizedLayoutMetadata(blockSize: 32, bytesPerBlock: 34, scaleLayout: .embeddedFp16Prefix)
    public static let float32 = QuantizedLayoutMetadata(blockSize: 1, bytesPerBlock: 4, scaleLayout: .none)
    public static let float16 = QuantizedLayoutMetadata(blockSize: 1, bytesPerBlock: 2, scaleLayout: .none)
}

/// A semantic representation of a parsed tensor preserving raw quantized layout without eager dequantization.
public struct QuantizedTensor: @unchecked Sendable {
    /// Identifier name of the tensor.
    public let name: String
    /// Dimension shape vector.
    public let shape: [Int]
    /// Quantization scheme (nil if standard unquantized float).
    public let scheme: QuantizationScheme?
    /// Raw binary buffer containing the packed weights and block headers.
    public let rawData: Data
    /// Scheme-specific block layout metadata.
    public let layoutMetadata: QuantizedLayoutMetadata

    public var elementCount: Int {
        shape.reduce(1, *)
    }

    public init(
        name: String,
        shape: [Int],
        scheme: QuantizationScheme?,
        rawData: Data,
        layoutMetadata: QuantizedLayoutMetadata
    ) {
        self.name = name
        self.shape = shape
        self.scheme = scheme
        self.rawData = rawData
        self.layoutMetadata = layoutMetadata
    }

    /// Extracts separated packed weights and scales buffers for GEMV execution.
    public func extractSeparatedBuffers() -> (weights: Data, scales: Data)? {
        guard let scheme = scheme else { return nil }
        let numBlocks = (elementCount + layoutMetadata.blockSize - 1) / layoutMetadata.blockSize
        switch scheme {
        case .q4_0:
            guard rawData.count >= numBlocks * 18 else { return nil }
            var scales = Data(capacity: numBlocks * 2)
            var weights = Data(capacity: numBlocks * 16)
            for b in 0..<numBlocks {
                let start = b * 18
                scales.append(rawData.subdata(in: start..<(start + 2)))
                weights.append(rawData.subdata(in: (start + 2)..<(start + 18)))
            }
            return (weights, scales)
        case .q8_0:
            guard rawData.count >= numBlocks * 34 else { return nil }
            var scales = Data(capacity: numBlocks * 2)
            var weights = Data(capacity: numBlocks * 32)
            for b in 0..<numBlocks {
                let start = b * 34
                scales.append(rawData.subdata(in: start..<(start + 2)))
                weights.append(rawData.subdata(in: (start + 2)..<(start + 34)))
            }
            return (weights, scales)
        default:
            return nil
        }
    }

    /// Explicit opt-in dequantization into a dense MLXArray (for debugging, testing, or CPU fallback).
    public func dequantizeToFloat() -> MLXArray {
        guard let scheme = self.scheme else {
            if layoutMetadata.bytesPerBlock == 2 {
                return MLXArray(rawData, shape, dtype: .float16)
            } else {
                return MLXArray(rawData, shape, dtype: .float32)
            }
        }

        let numBlocks = (elementCount + layoutMetadata.blockSize - 1) / layoutMetadata.blockSize
        switch scheme {
        case .q4_0:
            let floats = QuantizedTensor.dequantizeQ4_0(data: rawData, offset: 0, numBlocks: numBlocks, elementCount: elementCount)
            return MLXArray(floats, shape)
        case .q4_1:
            let floats = QuantizedTensor.dequantizeQ4_1(data: rawData, offset: 0, numBlocks: numBlocks, elementCount: elementCount)
            return MLXArray(floats, shape)
        case .q8_0:
            let floats = QuantizedTensor.dequantizeQ8_0(data: rawData, offset: 0, numBlocks: numBlocks, elementCount: elementCount)
            return MLXArray(floats, shape)
        case .awq4, .awq8:
            return MLX.zeros(shape, dtype: .float32)
        }
    }

    /// Access elements as typed array (convenience for testing).
    public func asArray<T: HasDType>(_ type: T.Type) -> [T] {
        dequantizeToFloat().asArray(type)
    }

    // MARK: - Dequantization Math Helpers

    private static func halfToFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let exp = UInt32((h >> 10) & 0x1F)
        let mant = UInt32(h & 0x03FF)

        if exp == 0 {
            if mant == 0 {
                return Float(bitPattern: sign)
            }
            var m = mant
            var e: Int32 = -14
            while (m & 0x0400) == 0 {
                m <<= 1
                e -= 1
            }
            let singleMant = (m & 0x03FF) << 13
            let singleExp = UInt32(bitPattern: (e + 127)) << 23
            return Float(bitPattern: sign | singleExp | singleMant)
        } else if exp == 0x1F {
            let singleExp = UInt32(0xFF) << 23
            let singleMant = mant << 13
            return Float(bitPattern: sign | singleExp | singleMant)
        } else {
            let singleExp = (exp - 15 + 127) << 23
            let singleMant = mant << 13
            return Float(bitPattern: sign | singleExp | singleMant)
        }
    }

    static func dequantizeQ4_0(data: Data, offset: Int, numBlocks: Int, elementCount: Int) -> [Float] {
        var result = [Float]()
        result.reserveCapacity(elementCount)
        var cur = offset

        for _ in 0..<numBlocks {
            let scaleRaw = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: cur, as: UInt16.self) }
            let scale = halfToFloat(scaleRaw.littleEndian)
            cur += 2

            for _ in 0..<16 {
                guard cur < data.count else { break }
                let byte = data[cur]
                cur += 1
                let low = Float(Int(byte & 0x0F) - 8) * scale
                let high = Float(Int(byte >> 4) - 8) * scale
                if result.count < elementCount { result.append(low) }
                if result.count < elementCount { result.append(high) }
            }
        }
        return result
    }

    static func dequantizeQ4_1(data: Data, offset: Int, numBlocks: Int, elementCount: Int) -> [Float] {
        var result = [Float]()
        result.reserveCapacity(elementCount)
        var cur = offset

        for _ in 0..<numBlocks {
            let scaleRaw = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: cur, as: UInt16.self) }
            let scale = halfToFloat(scaleRaw.littleEndian)
            cur += 2

            let minRaw = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: cur, as: UInt16.self) }
            let minVal = halfToFloat(minRaw.littleEndian)
            cur += 2

            for _ in 0..<16 {
                guard cur < data.count else { break }
                let byte = data[cur]
                cur += 1
                let low = Float(byte & 0x0F) * scale + minVal
                let high = Float(byte >> 4) * scale + minVal
                if result.count < elementCount { result.append(low) }
                if result.count < elementCount { result.append(high) }
            }
        }
        return result
    }

    static func dequantizeQ8_0(data: Data, offset: Int, numBlocks: Int, elementCount: Int) -> [Float] {
        var result = [Float]()
        result.reserveCapacity(elementCount)
        var cur = offset

        for _ in 0..<numBlocks {
            let scaleRaw = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: cur, as: UInt16.self) }
            let scale = halfToFloat(scaleRaw.littleEndian)
            cur += 2

            for _ in 0..<32 {
                guard cur < data.count else { break }
                let val = Int8(bitPattern: data[cur])
                cur += 1
                if result.count < elementCount {
                    result.append(Float(val) * scale)
                }
            }
        }
        return result
    }
}

extension Dictionary where Key == String, Value == QuantizedTensor {
    /// Dequantizes all contained tensors to dense MLXArrays.
    public func dequantized() -> [String: MLXArray] {
        mapValues { $0.dequantizeToFloat() }
    }
}
#endif
