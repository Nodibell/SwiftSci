import Foundation
import MLX

/// Inverse Protobuf binary reader for parsing tensor weight initializers from `.onnx` model files.
public struct ONNXWeightReader {
    
    /// Parses an ONNX model file Data payload into a dictionary of `[String: MLXArray]` weights.
    /// - Parameter data: Raw `.onnx` binary Protobuf data.
    /// - Returns: Dictionary of mapped tensor arrays keyed by parameter name.
    public static func parse(data: Data) -> [String: MLXArray] {
        var results = [String: MLXArray]()
        var index = 0
        
        while index < data.count {
            let (fieldNumber, wireType) = readTag(data: data, index: &index)
            if fieldNumber == 7 && wireType == 2 { // Field 7: GraphProto
                let graphData = readBytes(data: data, index: &index)
                parseGraph(data: graphData, results: &results)
            } else {
                skipField(wireType: wireType, data: data, index: &index)
            }
        }
        
        return results
    }

    private static func parseGraph(data: Data, results: inout [String: MLXArray]) {
        var index = 0
        while index < data.count {
            let (fieldNumber, wireType) = readTag(data: data, index: &index)
            if fieldNumber == 5 && wireType == 2 { // Field 5: initializer (repeated TensorProto)
                let tensorData = readBytes(data: data, index: &index)
                if let (name, array) = parseTensor(data: tensorData) {
                    results[name] = array
                }
            } else {
                skipField(wireType: wireType, data: data, index: &index)
            }
        }
    }

    private static func parseTensor(data: Data) -> (String, MLXArray)? {
        var index = 0
        var name = ""
        var dims = [Int]()
        var dataType = 1 // 1 = FLOAT, 11 = DOUBLE
        var floatValues = [Float]()
        var doubleValues = [Double]()
        var rawBytes: Data? = nil

        while index < data.count {
            let (fieldNumber, wireType) = readTag(data: data, index: &index)
            switch (fieldNumber, wireType) {
            case (1, 2): // name
                name = String(data: readBytes(data: data, index: &index), encoding: .utf8) ?? ""
            case (2, 0): // dims varint
                dims.append(Int(readVarint(data: data, index: &index)))
            case (2, 2): // dims packed varints
                let packed = readBytes(data: data, index: &index)
                var pIdx = 0
                while pIdx < packed.count {
                    dims.append(Int(readVarint(data: packed, index: &pIdx)))
                }
            case (3, 0): // data_type
                dataType = Int(readVarint(data: data, index: &index))
            case (4, 5): // float_data
                floatValues.append(readFloat32(data: data, index: &index))
            case (4, 2): // float_data packed
                let packed = readBytes(data: data, index: &index)
                var pIdx = 0
                while pIdx < packed.count {
                    floatValues.append(readFloat32(data: packed, index: &pIdx))
                }
            case (7, 1): // double_data
                doubleValues.append(readFloat64(data: data, index: &index))
            case (9, 2): // raw_data
                rawBytes = readBytes(data: data, index: &index)
            default:
                skipField(wireType: wireType, data: data, index: &index)
            }
        }

        guard !name.isEmpty else { return nil }

        if let raw = rawBytes {
            if dataType == 1 { // FLOAT
                let count = raw.count / MemoryLayout<Float>.size
                let floats = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                let shape = (dims.isEmpty || dims.reduce(1, *) != count) ? [count] : dims
                return (name, MLXArray(floats, shape))
            } else if dataType == 11 { // DOUBLE
                let count = raw.count / MemoryLayout<Double>.size
                let doubles = raw.withUnsafeBytes { Array($0.bindMemory(to: Double.self)) }
                let floats = doubles.map { Float($0) }
                let shape = (dims.isEmpty || dims.reduce(1, *) != count) ? [count] : dims
                return (name, MLXArray(floats, shape))
            }
        }

        if !floatValues.isEmpty {
            let shape = (dims.isEmpty || dims.reduce(1, *) != floatValues.count) ? [floatValues.count] : dims
            return (name, MLXArray(floatValues, shape))
        }

        if !doubleValues.isEmpty {
            let floats = doubleValues.map { Float($0) }
            let shape = (dims.isEmpty || dims.reduce(1, *) != doubleValues.count) ? [doubleValues.count] : dims
            return (name, MLXArray(floats, shape))
        }

        return nil
    }

    // MARK: - Low-level Protobuf Wire Helpers

    private static func readTag(data: Data, index: inout Int) -> (Int, Int) {
        let tag = readVarint(data: data, index: &index)
        let fieldNumber = Int(tag >> 3)
        let wireType = Int(tag & 0x07)
        return (fieldNumber, wireType)
    }

    private static func readVarint(data: Data, index: inout Int) -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        while index < data.count {
            let b = data[index]
            index += 1
            result |= UInt64(b & 0x7F) << shift
            if (b & 0x80) == 0 { break }
            shift += 7
        }
        return result
    }

    private static func readBytes(data: Data, index: inout Int) -> Data {
        guard index < data.count else { return Data() }
        let length = max(0, Int(readVarint(data: data, index: &index)))
        let start = min(data.count, index)
        let end = min(data.count, start + length)
        let sub = data.subdata(in: start..<end)
        index = end
        return sub
    }

    private static func readFloat32(data: Data, index: inout Int) -> Float {
        guard index + 4 <= data.count else {
            index = data.count
            return 0.0
        }
        let sub = data.subdata(in: index..<index + 4)
        index += 4
        return sub.withUnsafeBytes { $0.load(as: Float.self) }
    }

    private static func readFloat64(data: Data, index: inout Int) -> Double {
        guard index + 8 <= data.count else {
            index = data.count
            return 0.0
        }
        let sub = data.subdata(in: index..<index + 8)
        index += 8
        return sub.withUnsafeBytes { $0.load(as: Double.self) }
    }

    private static func skipField(wireType: Int, data: Data, index: inout Int) {
        switch wireType {
        case 0:
            _ = readVarint(data: data, index: &index)
        case 1:
            index = min(data.count, index + 8)
        case 2:
            let len = max(0, Int(readVarint(data: data, index: &index)))
            index = min(data.count, index + len)
        case 5:
            index = min(data.count, index + 4)
        default:
            index = data.count
        }
    }
}
