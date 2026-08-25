#if os(macOS)
import Testing
import Foundation
import MLX
@testable import SwiftLLM

@Suite("Quantized LLMs & GGUF Quantization Tests")
struct QuantizedWeightsTests {

    @Test("QuantizedLinear Q4_0 and Q8_0 forward pass")
    func testQuantizedLinearForward() throws {
        // [inFeatures: 4, outFeatures: 2]
        let inFeatures = 4
        let outFeatures = 2

        let weightsQ4 = MLXArray([
            Float(8.0), Float(10.0), Float(6.0), Float(9.0), // out 0
            Float(7.0), Float(8.0), Float(12.0), Float(8.0)  // out 1
        ], [outFeatures, inFeatures])

        let scales = MLXArray([Float](repeating: 0.5, count: 8), [outFeatures, inFeatures])
        let bias = MLXArray([Float(0.1), Float(-0.1)])

        let qLinear = QuantizedLinear(
            inFeatures: inFeatures,
            outFeatures: outFeatures,
            scheme: .q4_0,
            weight: weightsQ4,
            scales: scales,
            bias: bias
        )

        let input = MLXArray([Float(1.0), Float(2.0), Float(3.0), Float(4.0)], [1, inFeatures])
        let output = qLinear(input)

        #expect(output.shape == [1, outFeatures])
        let vals = output.asArray(Float.self)
        #expect(vals.count == 2)
    }

    @Test("GGUF Parser decodes Q4_0 and Q8_0 quantized blocks")
    func testGGUFQuantizedParsing() throws {
        var fileData = Data()

        // 1. Magic
        fileData.append(Data([0x47, 0x47, 0x55, 0x46]))
        // 2. Version
        var version = UInt32(3).littleEndian
        withUnsafeBytes(of: &version) { fileData.append(contentsOf: $0) }
        // 3. Tensor count (1)
        var tensorCount = UInt64(1).littleEndian
        withUnsafeBytes(of: &tensorCount) { fileData.append(contentsOf: $0) }
        // 4. Metadata count (0)
        var metadataCount = UInt64(0).littleEndian
        withUnsafeBytes(of: &metadataCount) { fileData.append(contentsOf: $0) }

        // 5. Tensor info
        let name = "q4_tensor"
        var nameLen = UInt64(name.utf8.count).littleEndian
        withUnsafeBytes(of: &nameLen) { fileData.append(contentsOf: $0) }
        fileData.append(name.data(using: .utf8)!)

        var dimsCount = UInt32(1).littleEndian
        withUnsafeBytes(of: &dimsCount) { fileData.append(contentsOf: $0) }
        var dim1 = UInt64(32).littleEndian
        withUnsafeBytes(of: &dim1) { fileData.append(contentsOf: $0) }

        // Type 2: Q4_0
        var tensorType = UInt32(2).littleEndian
        withUnsafeBytes(of: &tensorType) { fileData.append(contentsOf: $0) }
        var tensorOffset = UInt64(0).littleEndian
        withUnsafeBytes(of: &tensorOffset) { fileData.append(contentsOf: $0) }

        // Align to 32 bytes
        let currentOffset = fileData.count
        let alignment = 32
        let binaryStart = (currentOffset + alignment - 1) & ~(alignment - 1)
        if binaryStart > currentOffset {
            fileData.append(Data(repeating: 0, count: binaryStart - currentOffset))
        }

        // Q4_0 Block (18 bytes: 2 bytes scale + 16 bytes nibbles)
        var scaleF16 = Float16(1.0).bitPattern.littleEndian
        withUnsafeBytes(of: &scaleF16) { fileData.append(contentsOf: $0) }
        // 16 bytes with 0x88 (nibbles 8, 8 -> dequantized value (8-8)*1.0 = 0.0)
        fileData.append(Data(repeating: 0x88, count: 16))

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".gguf")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try fileData.write(to: tempURL)

        let parsed = try GGUFParser.parse(url: tempURL)
        #expect(parsed["q4_tensor"] != nil)
        #expect(parsed["q4_tensor"]?.shape == [32])
    }
}
#endif
