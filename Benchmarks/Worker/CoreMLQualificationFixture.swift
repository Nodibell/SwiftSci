import CoreML
import CryptoKit
import Foundation
import MLX
import SwiftML
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

// Shared artifact and execution fixture for controlled and trained-model inference.
struct CoreMLQualificationFixture {
    let name: String
    let rows: Int
    let width: Int
    let layers: [LayerWeights]
    let columns: [[Double]]
    let expected: [Double]
    var outputWidth: Int { layers.last!.outDim }
    var names: [String] { (0..<width).map { "feature_\($0)" } }

    init(name: String, rows: Int) throws {
        guard ["linear128", "mlp128", "mlp512"].contains(name), (1...1024).contains(rows) else {
            throw BenchmarkFailure("Unknown Core ML workload or rows outside 1...1024")
        }
        self.name = name
        self.rows = rows
        let width = name == "mlp512" ? 512 : 128
        self.width = width
        let columns: [[Double]] = (0..<width).map { column -> [Double] in
            (0..<rows).map { row -> Double in
                let numerator: Int = (row * 17 + column * 7) % 97 - 48
                return Double(numerator) / 16
            }
        }
        self.columns = columns
        let layers: [LayerWeights]
        if name == "linear128" {
            let weights: [Double] = (0..<width).map { Double($0 % 11 - 5) / 32 }
            layers = [LayerWeights(W: weights, b: [0.25], inDim: width, outDim: 1)]
        } else {
            var state: UInt64 = 0x5A17
            let dimensions: [Int] = [width, width, width, 1]
            var generated: [LayerWeights] = []
            for index in 0..<3 {
                let inputs = dimensions[index]
                let outputs = dimensions[index + 1]
                let weights: [Double] = (0..<(inputs * outputs)).map { _ -> Double in
                    state = state &* 6364136223846793005 &+ 1442695040888963407
                    return Double(Int((state >> 32) % 9) - 4) / 128
                }
                let biases = [Double](repeating: 0.0625, count: outputs)
                generated.append(LayerWeights(W: weights, b: biases, inDim: inputs, outDim: outputs))
            }
            layers = generated
        }
        self.layers = layers
        // Independent scalar Double oracle, outside every timed interval.
        var expected: [Double] = []
        expected.reserveCapacity(rows)
        for row in 0..<rows {
            var values: [Double] = columns.map { $0[row] }
            for (index, layer) in layers.enumerated() {
                var next = layer.b
                for i in 0..<layer.inDim {
                    for j in 0..<layer.outDim { next[j] += values[i] * layer.W[i * layer.outDim + j] }
                }
                if index < layers.count - 1 { next = next.map { max($0, 0) } }
                values = next
            }
            expected.append(values[0])
        }
        self.expected = expected
    }

    init(contentsOf url: URL, name: String, rows: Int) throws {
        let fixture = try JSONDecoder().decode(TrainedCoreMLFixture.self, from: Data(contentsOf: url))
        let columns = try fixture.inputColumns(relativeTo: url)
        let expected = try fixture.expectedValues(relativeTo: url)
        guard (1...8192).contains(rows), columns.count > 0, columns.count <= 1024,
              columns.allSatisfy({ $0.count == rows && $0.allSatisfy(\.isFinite) }),
              expected.allSatisfy(\.isFinite),
              !fixture.layers.isEmpty, fixture.layers.count <= 8 else {
            throw BenchmarkFailure("Invalid trained Core ML fixture dimensions or values")
        }
        var width = columns.count
        for layer in fixture.layers {
            guard layer.inDim == width, (1...1024).contains(layer.outDim),
                  layer.W.count == layer.inDim * layer.outDim, layer.b.count == layer.outDim,
                  layer.W.allSatisfy({ $0.isFinite && Float($0).isFinite }),
                  layer.b.allSatisfy({ $0.isFinite && Float($0).isFinite }) else {
                throw BenchmarkFailure("Invalid trained Core ML layer")
            }
            width = layer.outDim
        }
        guard expected.count == rows * width else { throw BenchmarkFailure("Trained output dimensions differ") }
        if let binary = fixture.expectedBinary {
            guard binary.rows == rows, binary.columns == width else {
                throw BenchmarkFailure("Binary output shape differs from trained model")
            }
        }
        self.name = name
        self.rows = rows
        self.width = columns.count
        self.layers = fixture.layers
        self.columns = columns
        self.expected = expected
    }

    func artifact(matrix: Bool) -> Data {
        if !matrix {
            return CoreMLExporter.exportBinaryMLPRegressor(inputNames: ["features"], outputName: "result", layers: layers)
        }
        // Exact rank-2 input/output and inner-product semantics follow Apple's schema:
        // https://apple.github.io/coremltools/mlmodel/Format/NeuralNetwork.html
        // Same Float32 weights as the public exporter; only the calling shape changes.
        var description = QualificationProto()
        description.bytes(1, feature("features", shape: [rows, width]))
        description.bytes(10, feature("result", shape: [rows, outputWidth]))
        var network = QualificationProto()
        network.integer(5, 1) // EXACT_ARRAY_MAPPING
        var inputName = "features"
        for (index, layer) in layers.enumerated() {
            let last = index == layers.count - 1
            let outputName = last ? "result" : "dense_\(index)"
            var weights: [Float] = []
            for j in 0..<layer.outDim {
                for i in 0..<layer.inDim { weights.append(Float(layer.W[i * layer.outDim + j])) }
            }
            var parameters = QualificationProto()
            parameters.integer(1, layer.inDim)
            parameters.integer(2, layer.outDim)
            parameters.integer(10, 1)
            parameters.bytes(20, floatWeights(weights))
            parameters.bytes(21, floatWeights(layer.b.map(Float.init)))
            var dense = QualificationProto()
            dense.string(1, "dense_\(index)")
            dense.string(2, inputName)
            dense.string(3, outputName)
            dense.bytes(140, parameters.data)
            network.bytes(1, dense.data)
            inputName = outputName
            if !last {
                var relu = QualificationProto()
                relu.bytes(10, Data())
                var activation = QualificationProto()
                activation.string(1, "relu_\(index)")
                activation.string(2, outputName)
                inputName = "activation_\(index)"
                activation.string(3, inputName)
                activation.bytes(130, relu.data)
                network.bytes(1, activation.data)
            }
        }
        var model = QualificationProto()
        model.integer(1, 4)
        model.bytes(2, description.data)
        model.bytes(500, network.data)
        return model.data
    }

    private func feature(_ name: String, shape: [Int]) -> Data {
        var array = QualificationProto()
        for size in shape { array.integer(1, size) }
        array.integer(2, 65600) // Double transport, Float32 weights.
        var type = QualificationProto()
        type.bytes(5, array.data)
        var feature = QualificationProto()
        feature.string(1, name)
        feature.bytes(3, type.data)
        return feature.data
    }

    private func floatWeights(_ values: [Float]) -> Data {
        var weights = QualificationProto()
        let data = values.withUnsafeBytes { Data($0) }
        weights.bytes(1, data)
        return weights.data
    }
}

// Minimal fixture encoder. The worker runs on little-endian Apple silicon.
private struct QualificationProto {
    var data = Data()
    mutating func varint(_ value: Int) {
        var value = UInt64(value)
        while value >= 128 { data.append(UInt8(value & 127) | 128); value >>= 7 }
        data.append(UInt8(value))
    }
    mutating func integer(_ field: Int, _ value: Int) { varint(field << 3); varint(value) }
    mutating func bytes(_ field: Int, _ value: Data) {
        varint((field << 3) | 2); varint(value.count); data.append(value)
    }
    mutating func string(_ field: Int, _ value: String) { bytes(field, Data(value.utf8)) }
}

// Benchmark-only fixed-weight MLX reference. It is not SwiftSci's public MLP estimator.
final class CoreMLQualificationMLX {
    let device: Device
    let weights: [MLXArray]
    let biases: [MLXArray]
    init(fixture: CoreMLQualificationFixture, gpu: Bool) {
        let device: Device = gpu ? .gpu : .cpu
        self.device = device
        (weights, biases) = Device.withDefaultDevice(device) {
            let weights = fixture.layers.map { MLXArray($0.W.map(Float.init), [$0.inDim, $0.outDim]) }
            let biases = fixture.layers.map { MLXArray($0.b.map(Float.init)) }
            for array in weights + biases { eval(array) }
            return (weights, biases)
        }
    }
    func predict(_ input: PreparedNumericBatch) throws -> [Double] {
        Device.withDefaultDevice(device) {
            let packed: [Float] = (0..<input.rowCount).flatMap { row -> [Float] in
                (0..<input.columnNames.count).map { Float(input[row, $0]!) }
            }
            var value = MLXArray(packed, [input.rowCount, input.columnNames.count])
            for index in weights.indices {
                value = matmul(value, weights[index]) + biases[index]
                if index < weights.count - 1 { value = maximum(value, 0) }
            }
            eval(value)
            StreamOrDevice.default.stream.synchronize()
            return value.asArray(Float.self).map(Double.init)
        }
    }
}

// Fixed-shape matrix experiment kept out of the public prepared-vector adapter.
final class CoreMLQualificationMatrix {
    let model: MLModel
    let rows: Int
    let width: Int
    let outputs: Int
    init(url: URL, units: MLComputeUnits, rows: Int, width: Int, outputs: Int) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        model = try MLModel(contentsOf: url, configuration: configuration)
        self.rows = rows
        self.width = width
        self.outputs = outputs
    }
    func predict(_ input: PreparedNumericBatch) throws -> [Double] {
        try autoreleasepool {
            let array = try MLMultiArray(shape: [NSNumber(value: rows), NSNumber(value: width)], dataType: .double)
            array.withUnsafeMutableBufferPointer(ofType: Double.self) { buffer, strides in
                for row in 0..<rows {
                    for column in 0..<width { buffer[row * strides[0] + column * strides[1]] = input[row, column]! }
                }
            }
            let provider = try MLDictionaryFeatureProvider(dictionary: ["features": MLFeatureValue(multiArray: array)])
            let output = try model.prediction(from: provider)
            guard let values = output.featureValue(for: "result")?.multiArrayValue,
                  values.shape == [NSNumber(value: rows), NSNumber(value: outputs)], values.dataType == .double else {
                throw BenchmarkFailure("Matrix fixture output shape or type changed")
            }
            return values.withUnsafeBufferPointer(ofType: Double.self) { buffer in
                let strides = values.strides.map(\.intValue)
                var result = [Double]()
                result.reserveCapacity(rows * outputs)
                for row in 0..<rows {
                    for column in 0..<outputs { result.append(buffer[row * strides[0] + column * strides[1]]) }
                }
                return result
            }
        }
    }
}

private struct TrainedCoreMLFixture: Decodable {
    let layers: [LayerWeights]
    let columns: [[Double]]?
    let expected: [Double]?
    let inputBinary: QualificationBinaryMatrix?
    let expectedBinary: QualificationBinaryMatrix?

    func inputColumns(relativeTo url: URL) throws -> [[Double]] {
        if let columns, inputBinary == nil { return columns }
        guard columns == nil, let inputBinary else { throw BenchmarkFailure("Ambiguous input fixture") }
        let values = try inputBinary.read(relativeTo: url)
        var result = [[Double]](repeating: [Double](repeating: 0, count: inputBinary.rows), count: inputBinary.columns)
        for row in 0..<inputBinary.rows {
            for column in 0..<inputBinary.columns { result[column][row] = values[row * inputBinary.columns + column] }
        }
        return result
    }

    func expectedValues(relativeTo url: URL) throws -> [Double] {
        if let expected, expectedBinary == nil { return expected }
        guard expected == nil, let expectedBinary else { throw BenchmarkFailure("Ambiguous output fixture") }
        return try expectedBinary.read(relativeTo: url)
    }
}

private struct QualificationBinaryMatrix: Decodable {
    let path: String
    let rows: Int
    let columns: Int
    let sha256: String

    func read(relativeTo fixture: URL) throws -> [Double] {
        guard (1...8192).contains(rows), (1...1024).contains(columns) else {
            throw BenchmarkFailure("Binary fixture exceeds bounded matrix dimensions")
        }
        let url = fixture.deletingLastPathComponent().appendingPathComponent(path)
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count == rows * columns * MemoryLayout<Double>.stride,
              SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == sha256 else {
            throw BenchmarkFailure("Binary fixture size or checksum differs")
        }
        return data.withUnsafeBytes { bytes in
            (0..<(rows * columns)).map { index in
                let bits = bytes.loadUnaligned(fromByteOffset: index * 8, as: UInt64.self)
                return Double(bitPattern: UInt64(littleEndian: bits))
            }
        }
    }
}
