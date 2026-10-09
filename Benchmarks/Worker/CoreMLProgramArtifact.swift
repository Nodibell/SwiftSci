import CoreML
import CryptoKit
import Foundation
import SwiftSciBenchmarkSupport

// Hash every source-package file, including weights. Compiled caches are deliberately excluded.
func coreMLPackageHash(_ url: URL) throws -> String {
    guard url.pathExtension == "mlpackage",
          let items = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey]) else {
        throw BenchmarkFailure("Expected an ML Program source package")
    }
    var files: [String: String] = [:]
    for case let file as URL in items {
        if try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
            let name = String(file.path.dropFirst(url.path.count + 1))
            files[name] = SHA256.hash(data: try Data(contentsOf: file)).map { String(format: "%02x", $0) }.joined()
        }
    }
    guard !files.isEmpty else { throw BenchmarkFailure("Empty ML Program package") }
    let data = try JSONSerialization.data(withJSONObject: files, options: [.sortedKeys])
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func programMatrixElementBytes(at url: URL, rows: Int, width: Int, outputs: Int) throws -> Int {
    let config = MLModelConfiguration()
    config.computeUnits = .cpuOnly
    let model = try MLModel(contentsOf: url, configuration: config)
    let description = model.modelDescription
    guard let input = description.inputDescriptionsByName["features"]?.multiArrayConstraint,
          let output = description.outputDescriptionsByName["result"]?.multiArrayConstraint,
          [.float16, .float32].contains(input.dataType), output.dataType == input.dataType,
          input.shape.map(\.intValue) == [rows, width],
          output.shape.map(\.intValue) == [rows, outputs] else {
        throw BenchmarkFailure("ML Program must match fixture dimensions with matching Float16 or Float32 matrix I/O")
    }
    return input.dataType == .float16 ? 2 : 4
}
