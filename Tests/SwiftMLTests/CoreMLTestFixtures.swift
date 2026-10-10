import CoreML
import Foundation
@testable import SwiftML

func withCompiledCoreML(_ data: Data, body: (URL) async throws -> Void) async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let source = directory.appendingPathComponent("Fixture.mlmodel")
    try data.write(to: source)
    let compiled = try await MLModel.compileModel(at: source)
    defer { try? FileManager.default.removeItem(at: compiled) }
    try await body(compiled)
}

// A ReLU fixture with selectable tensor type/rank, independent of the neural exporter.
// Field numbers follow Apple's Model, FeatureTypes, and NeuralNetwork protobuf schemas.
func coreMLReLUArtifact(shape: [Int], float32: Bool = true) -> Data {
    func feature(_ name: String) -> Data {
        var array = ProtobufWriter()
        for size in shape { array.writeVarintField(fieldNumber: 1, value: UInt64(size)) }
        array.writeVarintField(fieldNumber: 2, value: float32 ? 65568 : 65600)
        var type = ProtobufWriter()
        type.writeBytesField(fieldNumber: 5, bytes: array.data)
        var feature = ProtobufWriter()
        feature.writeStringField(fieldNumber: 1, value: name)
        feature.writeBytesField(fieldNumber: 3, bytes: type.data)
        return feature.data
    }
    var description = ProtobufWriter()
    description.writeBytesField(fieldNumber: 1, bytes: feature("features"))
    description.writeBytesField(fieldNumber: 10, bytes: feature("result"))
    var activation = ProtobufWriter()
    activation.writeBytesField(fieldNumber: 10, bytes: Data())
    var layer = ProtobufWriter()
    layer.writeStringField(fieldNumber: 1, value: "relu")
    layer.writeStringField(fieldNumber: 2, value: "features")
    layer.writeStringField(fieldNumber: 3, value: "result")
    layer.writeBytesField(fieldNumber: 130, bytes: activation.data)
    var network = ProtobufWriter()
    network.writeBytesField(fieldNumber: 1, bytes: layer.data)
    network.writeVarintField(fieldNumber: 5, value: 1)
    var model = ProtobufWriter()
    model.writeVarintField(fieldNumber: 1, value: 4)
    model.writeBytesField(fieldNumber: 2, bytes: description.data)
    model.writeBytesField(fieldNumber: 500, bytes: network.data)
    return model.data
}
