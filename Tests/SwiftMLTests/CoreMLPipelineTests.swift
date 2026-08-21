import XCTest
@testable import SwiftML

final class CoreMLPipelineTests: XCTestCase {

    func testPipelineClassifierSerialization() throws {
        // Create scaler binary
        let scalerData = CoreMLExporter.exportBinaryStandardScaler(
            name: "ScalerStage",
            inputNames: ["f1", "f2"],
            outputNames: ["scaled_f1", "scaled_f2"],
            shiftValues: [-10.0, -20.0],
            scaleValues: [0.5, 0.25]
        )
        XCTAssertGreaterThan(scalerData.count, 0)

        // Create logistic classifier binary
        let clfData = CoreMLExporter.exportBinaryLogisticModel(
            name: "ClfStage",
            inputNames: ["scaled_f1", "scaled_f2"],
            outputName: "label",
            weights: [1.2, -0.8],
            bias: 0.5,
            classLabels: [0, 1]
        )
        XCTAssertGreaterThan(clfData.count, 0)

        // Composite pipeline classifier
        let pipelineData = CoreMLExporter.exportBinaryPipelineClassifier(
            name: "ChainedClassificationPipeline",
            inputNames: ["f1", "f2"],
            outputName: "label",
            submodelsData: [scalerData, clfData],
            submodelNames: ["Scaler", "Classifier"],
            classLabels: [0, 1]
        )

        XCTAssertGreaterThan(pipelineData.count, scalerData.count + clfData.count)

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("PipelineClf_\(UUID().uuidString).mlmodel")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try CoreMLExporter.writePipelineClassifier(
            to: tempURL,
            inputNames: ["f1", "f2"],
            outputName: "label",
            submodelsData: [scalerData, clfData]
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempURL.path))
    }

    func testPipelineRegressorSerialization() throws {
        let scalerData = CoreMLExporter.exportBinaryStandardScaler(
            name: "ScalerStage",
            inputNames: ["sqft", "rooms"],
            shiftValues: [-1000.0, -3.0],
            scaleValues: [0.001, 0.333]
        )

        let regData = CoreMLExporter.exportBinaryLinearModel(
            name: "RegStage",
            inputNames: ["sqft", "rooms"],
            outputName: "price",
            weights: [200.0, 15000.0],
            bias: 50000.0
        )

        let pipelineData = CoreMLExporter.exportBinaryPipelineRegressor(
            name: "ChainedRegressionPipeline",
            inputNames: ["sqft", "rooms"],
            outputName: "price",
            submodelsData: [scalerData, regData]
        )
        XCTAssertGreaterThan(pipelineData.count, scalerData.count + regData.count)

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("PipelineReg_\(UUID().uuidString).mlmodel")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try CoreMLExporter.writePipelineRegressor(
            to: tempURL,
            inputNames: ["sqft", "rooms"],
            outputName: "price",
            submodelsData: [scalerData, regData]
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempURL.path))
    }

    func testMLPackageExport() throws {
        let linearData = CoreMLExporter.exportBinaryLinearModel(
            name: "SampleLinear",
            inputNames: ["x1"],
            outputName: "y",
            weights: [2.5],
            bias: 1.0
        )

        let packageURL = FileManager.default.temporaryDirectory.appendingPathComponent("SampleLinear_\(UUID().uuidString).mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try CoreMLExporter.writeMLPackage(
            modelData: linearData,
            to: packageURL,
            author: "SwiftSciTest",
            description: "Unit Test Package"
        )

        let manifestURL = packageURL.appendingPathComponent("Manifest.json")
        let modelURL = packageURL.appendingPathComponent("Data/com.apple.CoreML/model.mlmodel")

        XCTAssertTrue(FileManager.default.fileExists(atPath: manifestURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: modelURL.path))

        let manifestContent = try String(contentsOf: manifestURL)
        XCTAssertTrue(manifestContent.contains("SwiftSciTest"))
        XCTAssertTrue(manifestContent.contains("com.apple.CoreML/model.mlmodel"))
    }

    func testCoreMLExportableMLPackageConformance() async throws {
        let model = LinearRegression()
        let features = [[1.0], [2.0], [3.0], [4.0]]
        let targets = [2.0, 4.0, 6.0, 8.0]
        try await model.fit(features: features, targets: targets)

        let packageURL = FileManager.default.temporaryDirectory.appendingPathComponent("FittedModel_\(UUID().uuidString).mlpackage")
        defer { try? FileManager.default.removeItem(at: packageURL) }

        try await model.writeMLPackage(
            to: packageURL,
            featureNames: ["feature_x"],
            outputName: "target_y",
            author: "SwiftSciUnitTests",
            description: "Fitted Linear Regression"
        )

        XCTAssertTrue(FileManager.default.fileExists(atPath: packageURL.appendingPathComponent("Manifest.json").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: packageURL.appendingPathComponent("Data/com.apple.CoreML/model.mlmodel").path))
    }
}
