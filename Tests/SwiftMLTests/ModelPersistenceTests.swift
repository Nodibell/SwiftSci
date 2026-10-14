import Testing
import Foundation
@testable import SwiftML

@Suite("Model Persistence Roundtrip Tests")
struct ModelPersistenceTests {

    @Test("DecisionTreeClassifier save and load roundtrip preserves predictions and parameters")
    func testDecisionTreeClassifierPersistence() async throws {
        let features: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [2.0, 2.1],
            [8.0, 8.0], [8.5, 7.5], [9.0, 8.8]
        ]
        let targets: [Double] = [0, 0, 0, 1, 1, 1]

        let tree = DecisionTreeClassifier(maxDepth: 3, criterion: .entropy)
        try await tree.fit(features: features, targets: targets)
        let origPreds = try await tree.predict(features: features)

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("dt_clf_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try await tree.save(to: tempURL)
        #expect(FileManager.default.fileExists(atPath: tempURL.path))

        let loadedTree = try DecisionTreeClassifier.load(from: tempURL)
        let loadedPreds = try await loadedTree.predict(features: features)

        #expect(origPreds == loadedPreds)
        #expect(origPreds == [0, 0, 0, 1, 1, 1])
    }

    @Test("DecisionTreeRegressor save and load roundtrip preserves continuous predictions")
    func testDecisionTreeRegressorPersistence() async throws {
        let features: [[Double]] = [
            [1.0], [2.0], [3.0], [4.0], [5.0]
        ]
        let targets: [Double] = [2.0, 4.0, 6.0, 8.0, 10.0]

        let reg = DecisionTreeRegressor(maxDepth: 4)
        try await reg.fit(features: features, targets: targets)
        let origPreds = try await reg.predict(features: features)

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("dt_reg_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try await reg.save(to: tempURL)
        #expect(FileManager.default.fileExists(atPath: tempURL.path))

        let loadedReg = try DecisionTreeRegressor.load(from: tempURL)
        let loadedPreds = try await loadedReg.predict(features: features)

        #expect(origPreds.count == loadedPreds.count)
        for i in 0..<origPreds.count {
            #expect(abs(origPreds[i] - loadedPreds[i]) < 1e-6)
        }
    }

    @Test("RandomForestClassifier save and load roundtrip preserves ensemble predictions")
    func testRandomForestClassifierPersistence() async throws {
        let features: [[Double]] = [
            [0.1, 0.2], [0.2, 0.1], [0.3, 0.3],
            [5.0, 5.1], [5.2, 4.9], [4.8, 5.0]
        ]
        let targets: [Double] = [0, 0, 0, 1, 1, 1]

        let rf = RandomForestClassifier(nEstimators: 5, maxDepth: 3, randomState: 42)
        try await rf.fit(features: features, targets: targets)
        let origPreds = try await rf.predict(features: features)

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("rf_clf_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try await rf.save(to: tempURL)
        #expect(FileManager.default.fileExists(atPath: tempURL.path))

        let loadedRF = try RandomForestClassifier.load(from: tempURL)
        let loadedPreds = try await loadedRF.predict(features: features)

        #expect(origPreds == loadedPreds)
        #expect(origPreds == [0, 0, 0, 1, 1, 1])
    }

    @Test("RandomForestRegressor save and load roundtrip preserves ensemble predictions")
    func testRandomForestRegressorPersistence() async throws {
        let features: [[Double]] = [
            [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]
        ]
        let targets: [Double] = [3.0, 5.0, 7.0, 9.0]

        let rf = RandomForestRegressor(nEstimators: 5, maxDepth: 3, randomState: 42)
        try await rf.fit(features: features, targets: targets)
        let origPreds = try await rf.predict(features: features)

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("rf_reg_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try await rf.save(to: tempURL)
        #expect(FileManager.default.fileExists(atPath: tempURL.path))

        let loadedRF = try RandomForestRegressor.load(from: tempURL)
        let loadedPreds = try await loadedRF.predict(features: features)

        #expect(origPreds.count == loadedPreds.count)
        for i in 0..<origPreds.count {
            #expect(abs(origPreds[i] - loadedPreds[i]) < 1e-6)
        }
    }

    @Test("Unfitted model save throws SwiftMLError.notFitted")
    func testUnfittedModelSaveThrows() async {
        let unfittedTree = DecisionTreeClassifier()
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("unfitted_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        await #expect(throws: SwiftMLError.self) {
            try await unfittedTree.save(to: tempURL)
        }
    }
}
