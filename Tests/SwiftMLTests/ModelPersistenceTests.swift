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

    @Test("RandomForestClassifier roundtrip preserves arbitrary non-consecutive class labels and probabilities")
    func testRandomForestClassifierNonConsecutiveLabelsPersistence() async throws {
        let features: [[Double]] = [
            [0.1, 0.2], [0.2, 0.1],
            [5.0, 5.1], [5.2, 4.9],
            [10.0, 10.5], [10.2, 9.8]
        ]
        let targets: [Double] = [10.0, 10.0, 20.0, 20.0, 50.0, 50.0]

        let rf = RandomForestClassifier(nEstimators: 7, maxDepth: 4, randomState: 42)
        try await rf.fit(features: features, targets: targets)
        let origPreds = try await rf.predict(features: features)
        let origProbs = try await rf.predictProbability(features: features)
        let origClasses = await rf.classes

        #expect(origClasses == [10.0, 20.0, 50.0])
        #expect(origPreds == [10, 10, 20, 20, 50, 50])

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("rf_arbitrary_clf_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try await rf.save(to: tempURL)
        #expect(FileManager.default.fileExists(atPath: tempURL.path))

        let loadedRF = try RandomForestClassifier.load(from: tempURL)
        let loadedPreds = try await loadedRF.predict(features: features)
        let loadedProbs = try await loadedRF.predictProbability(features: features)
        let loadedClasses = await loadedRF.classes

        #expect(loadedClasses == [10.0, 20.0, 50.0])
        #expect(origPreds == loadedPreds)
        #expect(origProbs.count == loadedProbs.count)
        for i in 0..<origProbs.count {
            #expect(origProbs[i].count == loadedProbs[i].count)
            #expect(origProbs[i].count == 3)
            for j in 0..<origProbs[i].count {
                #expect(abs(origProbs[i][j] - loadedProbs[i][j]) < 1e-6)
            }
        }
    }

    @Test("DecisionTreeClassifier roundtrip preserves arbitrary non-consecutive class labels and probabilities")
    func testDecisionTreeClassifierNonConsecutiveLabelsPersistence() async throws {
        let features: [[Double]] = [
            [0.1, 0.2], [0.2, 0.1],
            [5.0, 5.1], [5.2, 4.9],
            [10.0, 10.5], [10.2, 9.8]
        ]
        let targets: [Double] = [10.0, 10.0, 20.0, 20.0, 50.0, 50.0]

        let dt = DecisionTreeClassifier(maxDepth: 4)
        try await dt.fit(features: features, targets: targets)
        let origPreds = try await dt.predict(features: features)
        let origClasses = await dt.classes

        #expect(origClasses == [10.0, 20.0, 50.0])
        #expect(origPreds == [10, 10, 20, 20, 50, 50])

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("dt_arbitrary_clf_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try await dt.save(to: tempURL)
        let loadedDT = try DecisionTreeClassifier.load(from: tempURL)
        let loadedPreds = try await loadedDT.predict(features: features)
        let loadedClasses = await loadedDT.classes

        #expect(loadedClasses == [10.0, 20.0, 50.0])
        #expect(origPreds == loadedPreds)
    }

    @Test("DecisionTreeRegressor and RandomForestRegressor preserve negative and continuous targets")
    func testRegressorsNegativeContinuousTargets() async throws {
        let features: [[Double]] = [
            [1.0], [2.0], [3.0], [4.0], [5.0]
        ]
        let targets: [Double] = [-15.5, -2.3, 0.0, 4.2, 19.8]

        let dtReg = DecisionTreeRegressor(maxDepth: 4)
        try await dtReg.fit(features: features, targets: targets)
        let dtOrigPreds = try await dtReg.predict(features: features)

        let rfReg = RandomForestRegressor(nEstimators: 5, maxDepth: 4, randomState: 42)
        try await rfReg.fit(features: features, targets: targets)
        let rfOrigPreds = try await rfReg.predict(features: features)

        let dtURL = FileManager.default.temporaryDirectory.appendingPathComponent("dt_reg_neg_\(UUID().uuidString).json")
        let rfURL = FileManager.default.temporaryDirectory.appendingPathComponent("rf_reg_neg_\(UUID().uuidString).json")
        defer {
            try? FileManager.default.removeItem(at: dtURL)
            try? FileManager.default.removeItem(at: rfURL)
        }

        try await dtReg.save(to: dtURL)
        try await rfReg.save(to: rfURL)

        let loadedDT = try DecisionTreeRegressor.load(from: dtURL)
        let loadedRF = try RandomForestRegressor.load(from: rfURL)

        let dtLoadedPreds = try await loadedDT.predict(features: features)
        let rfLoadedPreds = try await loadedRF.predict(features: features)

        for i in 0..<targets.count {
            #expect(abs(dtOrigPreds[i] - dtLoadedPreds[i]) < 1e-6)
            #expect(abs(rfOrigPreds[i] - rfLoadedPreds[i]) < 1e-6)
        }
    }

    @Test("Corrupted JSON data throws decoding error on load")
    func testCorruptedJSONPersistenceThrows() throws {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("corrupted_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let corruptedData = Data("{\"invalid\": \"payload\"}".utf8)
        try corruptedData.write(to: tempURL)

        #expect(throws: (any Error).self) {
            _ = try DecisionTreeClassifier.load(from: tempURL)
        }
        #expect(throws: (any Error).self) {
            _ = try DecisionTreeRegressor.load(from: tempURL)
        }
        #expect(throws: (any Error).self) {
            _ = try RandomForestClassifier.load(from: tempURL)
        }
        #expect(throws: (any Error).self) {
            _ = try RandomForestRegressor.load(from: tempURL)
        }
        #expect(throws: (any Error).self) {
            _ = try LinearRegression.load(from: tempURL)
        }
        #expect(throws: (any Error).self) {
            _ = try LogisticRegression.load(from: tempURL)
        }
    }

    @Test("Hyperparameters are correctly preserved across save and load")
    func testHyperparametersPreservedAcrossRoundtrip() async throws {
        let features: [[Double]] = [[1.0, 2.0], [2.0, 1.0], [5.0, 6.0], [6.0, 5.0]]
        let targets: [Double] = [1.0, 1.0, 2.0, 2.0]

        let rf = RandomForestClassifier(
            nEstimators: 12,
            maxDepth: 4,
            maxFeatures: 2,
            maxSamples: 3,
            minSamplesSplit: 2,
            criterion: .entropy,
            randomState: 777
        )
        try await rf.fit(features: features, targets: targets)

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("rf_params_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try await rf.save(to: tempURL)
        let loadedRF = try RandomForestClassifier.load(from: tempURL)

        #expect(await loadedRF.nEstimators == 12)
        #expect(await loadedRF.maxDepth == 4)
        #expect(await loadedRF.maxFeatures == 2)
        #expect(await loadedRF.maxSamples == 3)
        #expect(await loadedRF.minSamplesSplit == 2)
        #expect(await loadedRF.criterion == .entropy)
        #expect(await loadedRF.randomState == 777)
        #expect(await loadedRF.classes == [1.0, 2.0])
    }

    @Test("All unfitted models throw error when attempting to save")
    func testAllUnfittedModelsThrowOnSave() async {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("unfitted_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let dtc = DecisionTreeClassifier()
        let dtr = DecisionTreeRegressor()
        let rfc = RandomForestClassifier()
        let rfr = RandomForestRegressor()
        let lr = LinearRegression()
        let logr = LogisticRegression()

        await #expect(throws: SwiftMLError.self) {
            try await dtc.save(to: tempURL)
        }
        await #expect(throws: SwiftMLError.self) {
            try await dtr.save(to: tempURL)
        }
        await #expect(throws: SwiftMLError.self) {
            try await rfc.save(to: tempURL)
        }
        await #expect(throws: SwiftMLError.self) {
            try await rfr.save(to: tempURL)
        }
        await #expect(throws: SwiftMLError.self) {
            try await lr.save(to: tempURL)
        }
        await #expect(throws: SwiftMLError.self) {
            try await logr.save(to: tempURL)
        }
    }
}
