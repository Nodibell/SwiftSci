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

        let fileURL = URL(fileURLWithPath: "/tmp/test_linear.onnx")
        try binaryData.write(to: fileURL)
    }

    func testONNXExporterBinaryWithONNXRuntimeValidation() throws {
        let binaryData = ONNXExporter.exportBinaryONNX(
            name: "TestONNXBinaryModel",
            inputs: ["x"],
            output: "y",
            weights: [0.8, 0.2],
            bias: 0.1
        )

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("TestLinear_\(UUID().uuidString).onnx")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try binaryData.write(to: tempURL)

        // Run python3 verification with official onnx.checker and onnxruntime
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "python3", "-c",
            """
            import sys, onnx, onnxruntime as ort, numpy as np
            model_path = '\(tempURL.path)'
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            sess = ort.InferenceSession(model_path)
            x = np.array([[1.0, 2.0], [3.0, -1.0]], dtype=np.float64)
            res = sess.run(None, {'x': x})[0]
            expected = np.array([[1.0*0.8 + 2.0*0.2 + 0.1], [3.0*0.8 + (-1.0)*0.2 + 0.1]])
            if not np.allclose(res, expected):
                print(f"Mismatch: got {res}, expected {expected}")
                sys.exit(1)
            """
        ]
        let pipe = Pipe()
        process.standardError = pipe
        process.standardOutput = pipe
        do {
            try process.run()
            process.waitUntilExit()
            if process.terminationStatus != 0 {
                let errData = pipe.fileHandleForReading.readDataToEndOfFile()
                let errMsg = String(data: errData, encoding: .utf8) ?? ""
                XCTFail("Python onnx.checker or onnxruntime validation failed: \(errMsg)")
            }
        } catch {
            print("Skipping Python ONNX Runtime test: \(error)")
        }
    }
}
