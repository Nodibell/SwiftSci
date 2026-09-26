import Foundation
import Testing

@testable import SwiftSciBenchmarkSupport

struct ProtocolTests {
  @Test func wrongOutputsAndInvalidTimingsCannotPass() {
    #expect(throws: (any Error).self) {
      try BenchmarkSample(elapsed: 1, values: [2], expected: [1], atol: 0, rtol: 0)
    }
    #expect(throws: (any Error).self) {
      try BenchmarkSample(elapsed: -1, values: [1], expected: [1], atol: 0, rtol: 0)
    }
    #expect(throws: (any Error).self) {
      try BenchmarkSample(elapsed: 1, values: [.nan], expected: [1], atol: 1, rtol: 1)
    }
    #expect(throws: (any Error).self) {
      try BenchmarkSample(elapsed: 1, values: [], expected: [1], atol: 0, rtol: 0)
    }
  }
  @Test func unresolvedTimingsStillValidateOutputs() throws {
    for elapsed: Int64 in [0, 42, 999, 1000] {
      let sample = try BenchmarkSample(
        elapsed: elapsed, values: [1], expected: [1], atol: 0, rtol: 0)
      #expect(sample.validated && sample.elapsed_ns == elapsed)
      #expect(sample.timing_resolved == (elapsed >= 1000))
    }
    #expect(throws: (any Error).self) {
      try BenchmarkSample(elapsed: 0, values: [2], expected: [1], atol: 0, rtol: 0)
    }
  }
  @Test func toleranceAndBinaryIdentity() throws {
    let sample = try BenchmarkSample(
      elapsed: 42, values: [1.0], expected: [1.00001], atol: 0.0001, rtol: 0)
    #expect(sample.validated && sample.maximum_absolute_error > 0)
    #expect(
      sample.output_sha256 == "6c3c396ed6b5c36dcae172271f462051b1266b851e92df3deea8ac65478fd712")
  }
  @Test func unalignedAndTruncatedBinary() throws {
    let bytes = Data([0, 0, 0, 0, 0, 0, 240, 63])
    #expect(try decodeDoubles(bytes) == [1])
    #expect(throws: (any Error).self) { try decodeDoubles(bytes.dropLast()) }
  }
  @Test func corruptionIsRejected() throws {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: file) }
    try Data("bad".utf8).write(to: file)
    #expect(throws: (any Error).self) {
      try verifiedData(path: file.path, sha256: String(repeating: "0", count: 64))
    }
  }
}

struct UnivariateFixtureTests {
  @Test func variableHeadersAndScientificNotation() throws {
    let data = Data("Header\r\n\r\n1e-3\r\n -2.5\r\n".utf8)
    #expect(try decodeUnivariate(data, skipRows: 2, rows: 2) == [0.001, -2.5])
    #expect(try decodeUnivariate(Data("1\n2\n".utf8), skipRows: 0, rows: 2) == [1, 2])
  }
  @Test func malformedValuesAndCountsFail() {
    for text in ["1\n", "1\n2\n3\n", "NaN\n1\n", "inf\n1\n", "wrong\n1\n"] {
      #expect(throws: (any Error).self) {
        try decodeUnivariate(Data(text.utf8), skipRows: 0, rows: 2)
      }
    }
    let valid = Data("1\n2\n".utf8)
    #expect(throws: (any Error).self) { try decodeUnivariate(valid, skipRows: 99, rows: 2) }
    #expect(throws: (any Error).self) { try decodeUnivariate(valid, skipRows: -1, rows: 2) }
    #expect(throws: (any Error).self) { try decodeUnivariate(valid, skipRows: 0, rows: 1) }
    #expect(throws: (any Error).self) { try decodeUnivariate(Data([255]), skipRows: 0, rows: 2) }
  }
}
