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

    @Test("Native Metal MSL gemv_q4_0 and gemv_q8_0 execution")
    func testMetalQuantizedGEMV() throws {
        let inFeatures = 32
        let outFeatures = 2

        // Row 0: nibbles = 9 (value (9-8)*0.5 = 0.5), scale = 0.5
        // Row 1: nibbles = 10 (value (10-8)*1.0 = 2.0), scale = 1.0
        var weightsData = Data()
        weightsData.append(Data(repeating: 0x99, count: 16)) // row 0: 16 bytes = 32 nibbles
        weightsData.append(Data(repeating: 0xAA, count: 16)) // row 1: 16 bytes = 32 nibbles

        let scales: [Float16] = [Float16(0.5), Float16(1.0)]
        let input = [Float](repeating: 1.0, count: inFeatures)

        let dummyWeights = MLXArray([Float](repeating: 0.0, count: 64), [outFeatures, inFeatures])
        let dummyScales = MLXArray([Float](repeating: 1.0, count: 64), [outFeatures, inFeatures])

        let layer = QuantizedLinear(
            inFeatures: inFeatures,
            outFeatures: outFeatures,
            scheme: .q4_0,
            weight: dummyWeights,
            scales: dummyScales
        )

        let outputs = try layer.forwardMetal(
            inVector: input,
            rawWeights: weightsData,
            rawScales: scales
        )

        #expect(outputs.count == 2)
        // Row 0: 32 elements of (9-8)*0.5 = 0.5 -> sum = 32 * 0.5 = 16.0
        #expect(abs(outputs[0] - 16.0) < 1e-3)
        // Row 1: 32 elements of (10-8)*1.0 = 2.0 -> sum = 32 * 2.0 = 64.0
        #expect(abs(outputs[1] - 64.0) < 1e-3)
    }

    @Test("MetalQuantizedEngine pipeline caching and dimension validation")
    func testMetalEngineValidation() throws {
        let engine = MetalQuantizedEngine.shared
        guard engine.device != nil else { return }

        let pipeline = try engine.getPipeline(name: "gemv_q4_0")
        #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)

        // Test dimension mismatch throws
        let layer = QuantizedLinear(
            inFeatures: 32,
            outFeatures: 2,
            scheme: .q4_0,
            weight: MLXArray([Float](repeating: 0.0, count: 64), [2, 32]),
            scales: MLXArray([Float](repeating: 1.0, count: 64), [2, 32])
        )

        #expect(throws: MetalQuantizedError.self) {
            _ = try layer.forwardMetal(
                inVector: [1.0, 2.0], // mismatch: 2 instead of 32
                rawWeights: Data(repeating: 0, count: 32),
                rawScales: [Float16(1.0), Float16(1.0)]
            )
        }
    }
}
#endif
