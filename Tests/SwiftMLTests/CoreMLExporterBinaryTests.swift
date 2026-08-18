import XCTest
@testable import SwiftML
#if canImport(CoreML)
import CoreML
#endif

final class CoreMLExporterBinaryTests: XCTestCase {

    // MARK: - ProtobufWriter Unit Tests

    func testProtobufWriterVarint() {
        var writer = ProtobufWriter()
        writer.writeVarintField(fieldNumber: 1, value: 4)
        XCTAssertEqual(writer.data, Data([0x08, 0x04]))
    }

    func testProtobufWriterPackedDoubles() {
        var writer = ProtobufWriter()
        writer.writePackedDoublesField(fieldNumber: 1, values: [1.0])
        // Tag: field 1, wireType 2 -> 0x0A, length 8 -> 0x08, then 8 bytes of double
        XCTAssertEqual(writer.data.count, 2 + 8)
        XCTAssertEqual(writer.data[0], 0x0A)
        XCTAssertEqual(writer.data[1], 0x08)
    }

    // MARK: - Binary Linear Regression (GLMRegressor)

    func testExportBinaryLinearModel() throws {
        let data = CoreMLExporter.exportBinaryLinearModel(
            name: "PricePredictor",
            inputNames: ["sqft", "bedrooms"],
            outputName: "price",
            weights: [250.0, 15000.0],
            bias: 50000.0
        )

        XCTAssertFalse(data.isEmpty)
        // specificationVersion tag: 0x08 0x04 (specVersion = 4)
        XCTAssertEqual(data[0], 0x08)
        XCTAssertEqual(data[1], 0x04)

        #if canImport(CoreML) && os(macOS)
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("PricePredictor_\(UUID().uuidString).mlmodel")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try data.write(to: tempURL)

        if #available(macOS 14.0, *) {
            let compiledURL = try MLModel.compileModel(at: tempURL)
            defer { try? FileManager.default.removeItem(at: compiledURL) }
            let model = try MLModel(contentsOf: compiledURL)
            XCTAssertNotNil(model.modelDescription.inputDescriptionsByName["sqft"])
            XCTAssertNotNil(model.modelDescription.inputDescriptionsByName["bedrooms"])
            XCTAssertNotNil(model.modelDescription.outputDescriptionsByName["price"])

            // Numerical prediction test
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: ["sqft": 100.0, "bedrooms": 2.0])
            let prediction = try model.prediction(from: inputProvider)
            let predictedPrice = prediction.featureValue(for: "price")?.doubleValue ?? 0.0
            let expectedPrice = 250.0 * 100.0 + 15000.0 * 2.0 + 50000.0 // 105000.0
            XCTAssertEqual(predictedPrice, expectedPrice, accuracy: 1e-3)
        }
        #endif
    }

    // MARK: - Binary Logistic Regression (GLMClassifier)

    func testExportBinaryLogisticModel() throws {
        let data = CoreMLExporter.exportBinaryLogisticModel(
            name: "SpamDetector",
            inputNames: ["f1", "f2"],
            outputName: "label",
            weights: [0.5, -0.3],
            bias: 0.1
        )

        XCTAssertFalse(data.isEmpty)
        XCTAssertEqual(data[0], 0x08)
        XCTAssertEqual(data[1], 0x04)

        #if canImport(CoreML) && os(macOS)
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("SpamDetector_\(UUID().uuidString).mlmodel")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try data.write(to: tempURL)

        if #available(macOS 14.0, *) {
            let compiledURL = try MLModel.compileModel(at: tempURL)
            defer { try? FileManager.default.removeItem(at: compiledURL) }
            let model = try MLModel(contentsOf: compiledURL)
            XCTAssertNotNil(model.modelDescription.inputDescriptionsByName["f1"])
            XCTAssertNotNil(model.modelDescription.outputDescriptionsByName["label"])

            // Predict sample
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: ["f1": 2.0, "f2": 1.0])
            let prediction = try model.prediction(from: inputProvider)
            let predictedLabel = prediction.featureValue(for: "label")?.int64Value
            XCTAssertNotNil(predictedLabel)
        }
        #endif
    }

    // MARK: - Binary Decision Tree Export

    func testExportBinaryDecisionTreeClassifier() throws {
        let leafLeft = FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: 0.0, isLeaf: true, impurityGain: 0.0)
        let leafRight = FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: 1.0, isLeaf: true, impurityGain: 0.0)
        let root = FlatTreeNode(featureIndex: 0, threshold: 5.0, leftChild: 0, rightChild: 1, value: 0.0, isLeaf: false, impurityGain: 0.5)

        let state = DecisionTreeModelState(maxDepth: 1, minSamplesSplit: 2, nodes: [leafLeft, leafRight, root], numFeatures: 1)
        let data = CoreMLExporter.exportBinaryDecisionTreeClassifier(
            state: state,
            featureNames: ["feature_0"],
            outputName: "label"
        )

        XCTAssertFalse(data.isEmpty)
        XCTAssertEqual(data[0], 0x08)
        XCTAssertEqual(data[1], 0x04)

        #if canImport(CoreML) && os(macOS)
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("TreeClf_\(UUID().uuidString).mlmodel")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try data.write(to: tempURL)

        if #available(macOS 14.0, *) {
            let compiledURL = try MLModel.compileModel(at: tempURL)
            defer { try? FileManager.default.removeItem(at: compiledURL) }
            let model = try MLModel(contentsOf: compiledURL)
            XCTAssertNotNil(model.modelDescription.inputDescriptionsByName["feature_0"])
            XCTAssertNotNil(model.modelDescription.outputDescriptionsByName["label"])
        }
        #endif
    }

    func testExportBinaryDecisionTreeRegressor() throws {
        let leaf = FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: 42.0, isLeaf: true, impurityGain: 0.0)
        let state = DecisionTreeModelState(maxDepth: 0, minSamplesSplit: 2, nodes: [leaf], numFeatures: 1)
        let data = CoreMLExporter.exportBinaryDecisionTreeRegressor(
            state: state,
            featureNames: ["x"],
            outputName: "target"
        )

        XCTAssertFalse(data.isEmpty)

        #if canImport(CoreML) && os(macOS)
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("TreeReg_\(UUID().uuidString).mlmodel")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try data.write(to: tempURL)

        if #available(macOS 14.0, *) {
            let compiledURL = try MLModel.compileModel(at: tempURL)
            defer { try? FileManager.default.removeItem(at: compiledURL) }
            let model = try MLModel(contentsOf: compiledURL)
            XCTAssertNotNil(model.modelDescription.inputDescriptionsByName["x"])
            XCTAssertNotNil(model.modelDescription.outputDescriptionsByName["target"])

            let inputProvider = try MLDictionaryFeatureProvider(dictionary: ["x": 10.0])
            let prediction = try model.prediction(from: inputProvider)
            let predictedTarget = prediction.featureValue(for: "target")?.doubleValue
            XCTAssertEqual(predictedTarget, 42.0)
        }
        #endif
    }

    // MARK: - Binary Random Forest Export

    func testExportBinaryRandomForestClassifier() throws {
        let leaf0 = FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: 0.0, isLeaf: true, impurityGain: 0.0)
        let leaf1 = FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: 1.0, isLeaf: true, impurityGain: 0.0)
        let state = RandomForestModelState(nEstimators: 2, maxDepth: 1, minSamplesSplit: 2, trees: [[leaf0], [leaf1]], numFeatures: 1)

        let data = CoreMLExporter.exportBinaryRandomForestClassifier(
            state: state,
            featureNames: ["feat"],
            outputName: "class"
        )
        XCTAssertFalse(data.isEmpty)

        #if canImport(CoreML) && os(macOS)
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("ForestClf_\(UUID().uuidString).mlmodel")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try data.write(to: tempURL)

        if #available(macOS 14.0, *) {
            let compiledURL = try MLModel.compileModel(at: tempURL)
            defer { try? FileManager.default.removeItem(at: compiledURL) }
            let model = try MLModel(contentsOf: compiledURL)
            XCTAssertNotNil(model.modelDescription.inputDescriptionsByName["feat"])
            XCTAssertNotNil(model.modelDescription.outputDescriptionsByName["class"])
        }
        #endif
    }

    func testExportBinaryRandomForestRegressor() throws {
        let leaf0 = FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: 10.0, isLeaf: true, impurityGain: 0.0)
        let leaf1 = FlatTreeNode(featureIndex: -1, threshold: 0, leftChild: -1, rightChild: -1, value: 20.0, isLeaf: true, impurityGain: 0.0)
        let state = RandomForestModelState(nEstimators: 2, maxDepth: 1, minSamplesSplit: 2, trees: [[leaf0], [leaf1]], numFeatures: 1)

        let data = CoreMLExporter.exportBinaryRandomForestRegressor(
            state: state,
            featureNames: ["feat"],
            outputName: "prediction"
        )
        XCTAssertFalse(data.isEmpty)

        #if canImport(CoreML) && os(macOS)
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("ForestReg_\(UUID().uuidString).mlmodel")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try data.write(to: tempURL)

        if #available(macOS 14.0, *) {
            let compiledURL = try MLModel.compileModel(at: tempURL)
            defer { try? FileManager.default.removeItem(at: compiledURL) }
            let model = try MLModel(contentsOf: compiledURL)
            XCTAssertNotNil(model.modelDescription.inputDescriptionsByName["feat"])
            XCTAssertNotNil(model.modelDescription.outputDescriptionsByName["prediction"])

            let inputProvider = try MLDictionaryFeatureProvider(dictionary: ["feat": 5.0])
            let prediction = try model.prediction(from: inputProvider)
            let predictedVal = prediction.featureValue(for: "prediction")?.doubleValue ?? 0.0
            // Average of 10.0 and 20.0 = 15.0
            XCTAssertEqual(predictedVal, 15.0, accuracy: 1e-3)
        }
        #endif
    }

    // MARK: - CoreMLExportable Protocol Conformances

    func testDecisionTreeClassifierConformance() async throws {
        let tree = DecisionTreeClassifier(maxDepth: 3)
        do {
            _ = try await tree.exportCoreML(featureNames: ["f1"])
            XCTFail("Should throw modelNotFitted before training")
        } catch let error as SwiftMLError {
            XCTAssertEqual(error, .modelNotFitted)
        }

        let X = [[1.0], [2.0], [10.0], [11.0]]
        let y: [Double] = [0.0, 0.0, 1.0, 1.0]
        try await tree.fit(features: X, targets: y)

        let data = try await tree.exportCoreML(featureNames: ["feature_x"], outputName: "class_label")
        XCTAssertFalse(data.isEmpty)
    }

    func testDecisionTreeRegressorConformance() async throws {
        let reg = DecisionTreeRegressor(maxDepth: 2)
        let X = [[1.0], [2.0], [3.0], [4.0]]
        let y = [10.0, 20.0, 30.0, 40.0]
        try await reg.fit(features: X, targets: y)

        let data = try await reg.exportCoreML(featureNames: ["x"], outputName: "y")
        XCTAssertFalse(data.isEmpty)
    }

    func testRandomForestClassifierConformance() async throws {
        let rf = try RandomForestClassifier(nEstimators: 5, maxDepth: 3)
        let X = [[1.0, 1.0], [2.0, 2.0], [8.0, 8.0], [9.0, 9.0]]
        let y: [Double] = [0.0, 0.0, 1.0, 1.0]
        try await rf.fit(features: X, targets: y)

        let data = try await rf.exportCoreML(featureNames: ["f1", "f2"], outputName: "label")
        XCTAssertFalse(data.isEmpty)

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("RFClf_\(UUID().uuidString).mlmodel")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try await rf.writeCoreML(to: tempURL, featureNames: ["f1", "f2"], outputName: "label")
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempURL.path))
    }

    func testRandomForestRegressorConformance() async throws {
        let rf = try RandomForestRegressor(nEstimators: 5, maxDepth: 3)
        let X = [[1.0], [2.0], [3.0], [4.0]]
        let y = [5.0, 10.0, 15.0, 20.0]
        try await rf.fit(features: X, targets: y)

        let data = try await rf.exportCoreML(featureNames: ["x"], outputName: "y_pred")
        XCTAssertFalse(data.isEmpty)

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("RFReg_\(UUID().uuidString).mlmodel")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try await rf.writeCoreML(to: tempURL, featureNames: ["x"], outputName: "y_pred")
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempURL.path))
    }

    #if os(macOS)
    func testLinearRegressionConformance() async throws {
        let reg = LinearRegression()
        let X = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]]
        let y = [3.0, 6.0, 9.0, 12.0, 15.0]
        try await reg.fit(features: X, targets: y)

        let data = try await reg.exportCoreML(featureNames: ["x1", "x2"], outputName: "y_pred")
        XCTAssertFalse(data.isEmpty)

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("LinearReg_\(UUID().uuidString).mlmodel")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try await reg.writeCoreML(to: tempURL, featureNames: ["x1", "x2"], outputName: "y_pred")
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempURL.path))
    }

    func testLogisticRegressionConformance() async throws {
        let clf = LogisticRegression()
        let X = [[1.0, 1.0], [2.0, 1.5], [8.0, 8.0], [9.0, 8.5]]
        let y: [Double] = [0.0, 0.0, 1.0, 1.0]
        try await clf.fit(features: X, targets: y)

        let data = try await clf.exportCoreML(featureNames: ["x1", "x2"], outputName: "is_positive")
        XCTAssertFalse(data.isEmpty)

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("LogisticReg_\(UUID().uuidString).mlmodel")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        try await clf.writeCoreML(to: tempURL, featureNames: ["x1", "x2"], outputName: "is_positive")
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempURL.path))
    }
    #endif
}
