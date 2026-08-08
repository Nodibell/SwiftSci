import XCTest
@testable import SwiftML

final class ExporterTests: XCTestCase {

    func testCoreMLExporterJSON() throws {
        let data = try CoreMLExporter.exportLinearModel(
            name: "TestLinearModel",
            inputNames: ["feature_1", "feature_2"],
            outputName: "prediction",
            weights: [1.5, -2.0],
            bias: 0.5
        )

        XCTAssertFalse(data.isEmpty)
        let jsonStr = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(jsonStr.contains("TestLinearModel"))
        XCTAssertTrue(jsonStr.contains("feature_1"))
    }

    func testONNXExporterJSON() throws {
        let data = try ONNXExporter.exportLinearONNX(
            name: "TestONNXModel",
            inputs: ["x1", "x2"],
            output: "y",
            weights: [0.8, 0.2],
            bias: 0.1
        )

        XCTAssertFalse(data.isEmpty)
        let jsonStr = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(jsonStr.contains("TestONNXModel"))
        XCTAssertTrue(jsonStr.contains("LinearRegressor"))
    }

    func testONNXExporterBinary() throws {
        let binaryData = ONNXExporter.exportBinaryONNX(
            name: "TestONNXBinaryModel",
            inputs: ["x1", "x2"],
            output: "y",
            weights: [0.8, 0.2],
            bias: 0.1
        )

        XCTAssertFalse(binaryData.isEmpty)
        // Verify IR version tag 0x08 0x08
        XCTAssertEqual(binaryData[0], 0x08)
        XCTAssertEqual(binaryData[1], 0x08)
    }
}
