import Foundation
import MLX
import Testing
@testable import SwiftVision

// Standard ONNX TensorProto fixtures, verified with the official Python parser.
private let tensorWireFixtures: [Data] = [
    "OhoqGAgBCAIQASIIAABAPwAAgD5CBnRlbnNvcg==",
    "OhoqGAgBCAIQAUIGdGVuc29ySggAAEA/AACAPg==",
    "OiIqIAgBCAIQC0IGdGVuc29yUhAAAAAAAADoPwAAAAAAANA/",
    "OiIqIAgBCAIQC0IGdGVuc29yShAAAAAAAADoPwAAAAAAANA/",
    "OhoqGAoCAQIQAUIGdGVuc29yJQAAQD8lAACAPg==",
    "OiIqIAoCAQIQC0IGdGVuc29yUQAAAAAAAOg/UQAAAAAAANA/",
].map { encoded -> Data in Data(base64Encoded: encoded)! }

@Suite("ONNX tensor wire compatibility")
struct ONNXWeightReaderWireTests {
    @Test(arguments: 0..<6)
    func standardFieldsPreserveNamesShapesAndValues(fixture: Int) throws {
        let weights = ONNXWeightReader.parse(data: tensorWireFixtures[fixture])
        let array = try #require(weights["tensor"])
        #expect(weights.count == 1)
        #expect(array.shape == [1, 2])
        let expected: [Float] = [0.75, 0.25]
        #expect(array.asArray(Float.self).map(\.bitPattern) == expected.map(\.bitPattern))
    }
}
