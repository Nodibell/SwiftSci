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

    // MARK: - Python Environment Helper

    private func ensurePythonONNXEnvironment() throws {
        let probe = Process()
        probe.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        probe.arguments = ["python3", "-c", "import onnx, onnxruntime, numpy"]
        let errPipe = Pipe()
        probe.standardError = errPipe
        do {
            try probe.run()
            probe.waitUntilExit()
            if probe.terminationStatus != 0 {
                if ProcessInfo.processInfo.environment["STRICT_ONNX_CI"] != nil {
                    let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                    let msg = String(data: errData, encoding: .utf8) ?? ""
                    XCTFail("Python onnx or onnxruntime missing with STRICT_ONNX_CI enabled: \(msg)")
                }
                throw XCTSkip("Skipping: python3 with onnx, onnxruntime, or numpy is not available.")
            }
        } catch let skip as XCTSkip {
            throw skip
        } catch {
            if ProcessInfo.processInfo.environment["STRICT_ONNX_CI"] != nil {
                XCTFail("Failed to execute python3 with STRICT_ONNX_CI enabled: \(error)")
            }
            throw XCTSkip("Skipping: cannot run python3 (\(error)).")
        }
    }

    private func runPythonONNXValidation(modelURL: URL, script: String) throws {
        try ensurePythonONNXEnvironment()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["python3", "-c", script]
        let pipe = Pipe()
        process.standardError = pipe
        process.standardOutput = pipe
        try process.run()
        process.waitUntilExit()
        if process.terminationStatus != 0 {
            let errData = pipe.fileHandleForReading.readDataToEndOfFile()
            let errMsg = String(data: errData, encoding: .utf8) ?? ""
            XCTFail("Python onnx.checker or onnxruntime validation failed: \(errMsg)")
        }
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

        try runPythonONNXValidation(modelURL: tempURL, script: """
            import sys, onnx, onnxruntime as ort, numpy as np
            model_path = '\(tempURL.path)'
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            sess = ort.InferenceSession(model_path)
            x = np.array([[1.0, 2.0], [3.0, -1.0]], dtype=np.float64)
            res = sess.run(None, {'x': x})[0]
            expected = np.array([[1.0*0.8 + 2.0*0.2 + 0.1], [3.0*0.8 + (-1.0)*0.2 + 0.1]])
            if not np.allclose(res, expected, atol=1e-6):
                print(f"Mismatch: got {res}, expected {expected}")
                sys.exit(1)
            """)
    }

    func testONNXExporterSingleFeatureWithONNXRuntime() throws {
        let binaryData = ONNXExporter.exportBinaryONNX(
            name: "SingleFeatureLinearONNX",
            inputs: ["x"],
            output: "y",
            weights: [2.5],
            bias: -1.0
        )

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("TestSingleFeature_\(UUID().uuidString).onnx")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try binaryData.write(to: tempURL)

        try runPythonONNXValidation(modelURL: tempURL, script: """
            import sys, onnx, onnxruntime as ort, numpy as np
            model_path = '\(tempURL.path)'
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            sess = ort.InferenceSession(model_path)
            x = np.array([[0.0], [2.0], [-3.0]], dtype=np.float64)
            res = sess.run(None, {'x': x})[0]
            expected = np.array([[-1.0], [4.0], [-8.5]], dtype=np.float64)
            if not np.allclose(res, expected, atol=1e-6):
                print(f"Mismatch: got {res}, expected {expected}")
                sys.exit(1)
            """)
    }

    func testONNXExporterMultiFeatureBatchWithONNXRuntime() throws {
        let weights = [1.2, -0.5, 0.0, 3.1, -2.4]
        let bias = 0.75
        let binaryData = ONNXExporter.exportBinaryONNX(
            name: "MultiFeatureLinearONNX",
            inputs: ["features"],
            output: "target",
            weights: weights,
            bias: bias
        )

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("TestMultiFeature_\(UUID().uuidString).onnx")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try binaryData.write(to: tempURL)

        try runPythonONNXValidation(modelURL: tempURL, script: """
            import sys, onnx, onnxruntime as ort, numpy as np
            model_path = '\(tempURL.path)'
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            sess = ort.InferenceSession(model_path)
            x = np.array([
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [-1.0, 0.0, 1.0, -2.0, 3.0],
                [0.5, -0.5, 2.0, 1.5, -1.0],
                [10.0, -10.0, 0.0, 5.0, 2.0]
            ], dtype=np.float64)
            res = sess.run(None, {'features': x})[0]
            w = np.array([1.2, -0.5, 0.0, 3.1, -2.4], dtype=np.float64).reshape(-1, 1)
            b = 0.75
            expected = np.matmul(x, w) + b
            if not np.allclose(res, expected, atol=1e-6):
                print(f"Mismatch: got {res}, expected {expected}")
                sys.exit(1)
            """)
    }

    func testONNXExporterLinearRegressionEndToEndParity() async throws {
        let features = [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0],
            [5.0, 6.0]
        ]
        let targets = [5.0, 4.0, 11.0, 10.0, 17.0]

        let model = LinearRegression()
        try await model.fit(features: features, targets: targets)
        let (weightsOpt, biasOpt) = await model.getWeightsAndBias()
        guard let weights = weightsOpt, let bias = biasOpt else {
            XCTFail("Model not fitted")
            return
        }

        let binaryData = ONNXExporter.exportBinaryONNX(
            name: "FittedLinearRegression",
            inputs: ["input_tensor"],
            output: "predictions",
            weights: weights,
            bias: bias
        )

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("TestEndToEnd_\(UUID().uuidString).onnx")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try binaryData.write(to: tempURL)

        let testInput = [
            [2.5, 3.5],
            [6.0, 5.0]
        ]
        let swiftPreds = try await model.predict(features: testInput)

        try runPythonONNXValidation(modelURL: tempURL, script: """
            import sys, onnx, onnxruntime as ort, numpy as np
            model_path = '\(tempURL.path)'
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            sess = ort.InferenceSession(model_path)
            x = np.array([[2.5, 3.5], [6.0, 5.0]], dtype=np.float64)
            res = sess.run(None, {'input_tensor': x})[0].flatten()
            expected = np.array([\(swiftPreds.map { String($0) }.joined(separator: ", "))], dtype=np.float64)
            if not np.allclose(res, expected, atol=1e-5):
                print(f"Mismatch: got {res}, expected {expected}")
                sys.exit(1)
            """)
    }
}
