import Testing
import Foundation
import MLX
@testable import SwiftVision

@Suite("ONNXWeightReader Tests")
struct ONNXWeightReaderTests {

    @Test("Empty data returns empty weight dictionary")
    func testEmptyData() {
        let weights = ONNXWeightReader.parse(data: Data())
        #expect(weights.isEmpty)
    }

    @Test("Corrupted data does not crash and returns parsed or empty weights")
    func testCorruptedData() {
        let garbage = Data([0xFF, 0xFE, 0xFD, 0x00, 0x01, 0x02])
        let weights = ONNXWeightReader.parse(data: garbage)
        #expect(weights.isEmpty)
    }

    @Test("Synthesized minimal ONNX graph initializer is parsed correctly")
    func testSynthesizedTensorInitializer() {
        var tensorData = Data()
        // Name (field 1, wireType 2)
        let nameBytes = "conv1.weight".data(using: .utf8)!
        tensorData.append(contentsOf: [1 << 3 | 2, UInt8(nameBytes.count)])
        tensorData.append(nameBytes)

        // Dims (field 2, wireType 0): 2
        tensorData.append(contentsOf: [2 << 3 | 0, 2])

        // Data type (field 3, wireType 0): 1 (FLOAT)
        tensorData.append(contentsOf: [3 << 3 | 0, 1])

        // Float data (field 4, wireType 5): 0.75f and 0.25f
        var f1: Float = 0.75
        var f2: Float = 0.25
        withUnsafeBytes(of: &f1) { tensorData.append(contentsOf: [4 << 3 | 5] + Array($0)) }
        withUnsafeBytes(of: &f2) { tensorData.append(contentsOf: [4 << 3 | 5] + Array($0)) }

        // Wrap into GraphProto (field 5 in Graph = initializer, wireType 2)
        var graphData = Data()
        graphData.append(contentsOf: [5 << 3 | 2, UInt8(tensorData.count)])
        graphData.append(tensorData)

        // Wrap into ModelProto (field 7 in Model = graph, wireType 2)
        var modelData = Data()
        modelData.append(contentsOf: [7 << 3 | 2, UInt8(graphData.count)])
        modelData.append(graphData)

        let weights = ONNXWeightReader.parse(data: modelData)
        #expect(weights["conv1.weight"] != nil)
        if let arr = weights["conv1.weight"] {
            #expect(arr.shape == [2])
            let vals = arr.asArray(Float.self)
            #expect(vals.count == 2)
            #expect(abs(vals[0] - 0.75) < 1e-5)
            #expect(abs(vals[1] - 0.25) < 1e-5)
        }
    }
}
