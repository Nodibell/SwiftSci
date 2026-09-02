import Testing
import Foundation
@testable import SwiftML

@Suite("DecisionTree Tests")
struct DecisionTreeTests {

    // MARK: Classifier

    @Test("DecisionTreeClassifier classifies linearly separable data")
    func testLinearSeparable() async throws {
        let features: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [2.0, 2.1],   // class 0
            [8.0, 8.0], [8.5, 7.5], [9.0, 8.8]    // class 1
        ]
        let targets: [Double] = [0, 0, 0, 1, 1, 1]

        let tree = DecisionTreeClassifier(maxDepth: 3)
        try await tree.fit(features: features, targets: targets)
        let preds = try await tree.predict(features: features)

        #expect(preds == [0, 0, 0, 1, 1, 1])

        let importances = await tree.featureImportances
        #expect(importances != nil)
        #expect(importances!.count == 2)
        let sumImp = importances!.reduce(0, +)
        #expect(abs(sumImp - 1.0) < 1e-5)
    }

    @Test("RandomForestClassifier computes feature importances")
    func testRandomForestFeatureImportances() async throws {
        let features: [[Double]] = [
            [1.0, 100.0], [1.5, 105.0], [2.0, 102.0],
            [8.0, 101.0], [8.5, 103.0], [9.0, 104.0]
        ]
        let targets: [Double] = [0, 0, 0, 1, 1, 1]

        let rf = try RandomForestClassifier(nEstimators: 10, maxDepth: 3)
        try await rf.fit(features: features, targets: targets)

        let importances = await rf.featureImportances
        #expect(importances != nil)
        #expect(importances!.count == 2)
        let sumImp = importances!.reduce(0, +)
        #expect(abs(sumImp - 1.0) < 1e-5)
    }

    @Test("DecisionTreeClassifier classifies XOR-like non-linear data")
    func testXOR() async throws {
        // XOR: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
        let features: [[Double]] = [[0, 0], [0, 1], [1, 0], [1, 1]]
        let targets: [Double]   = [0, 1, 1, 0]

        let tree = DecisionTreeClassifier(maxDepth: 5)
        try await tree.fit(features: features, targets: targets)
        let preds = try await tree.predict(features: features)

        #expect(preds == [0, 1, 1, 0])
    }

    @Test("DecisionTreeClassifier throws on empty input")
    func testClassifierEmptyInput() async throws {
        let tree = DecisionTreeClassifier()
        await #expect(throws: MLError.self) {
            try await tree.fit(features: [], targets: [])
        }
    }

    @Test("DecisionTreeClassifier throws when not fitted")
    func testClassifierNotFitted() async throws {
        let tree = DecisionTreeClassifier()
        await #expect(throws: MLError.self) {
            _ = try await tree.predict(features: [[1.0, 2.0]])
        }
    }

    // MARK: Regressor

    @Test("DecisionTreeRegressor fits a piecewise linear function")
    func testRegressionConvergence() async throws {
        // y = x (simple increasing function)
        let features: [[Double]] = (1...10).map { [Double($0)] }
        let targets: [Double]    = (1...10).map { Double($0) }

        let tree = DecisionTreeRegressor(maxDepth: 5)
        try await tree.fit(features: features, targets: targets)
        let preds = try await tree.predict(features: features)

        // MSE should be very small for training data
        let mse = zip(targets, preds).map { pow($0 - $1, 2) }.reduce(0, +) / Double(targets.count)
        #expect(mse < 1.0)
    }

    @Test("DecisionTreeRegressor throws when not fitted")
    func testRegressorNotFitted() async throws {
        let tree = DecisionTreeRegressor()
        await #expect(throws: MLError.self) {
            _ = try await tree.predict(features: [[1.0]])
        }
    }
}

@Suite("RandomForest Tests")
struct RandomForestTests {

    @Test("RandomForestClassifier classifies separable data with high accuracy")
    func testRFClassification() async throws {
        var features = [[Double]]()
        var targets = [Double]()
        for i in 0..<50 {
            features.append([Double(i), Double(i) * 0.5])
            targets.append(i < 25 ? 0 : 1)
        }

        let rf = try RandomForestClassifier(nEstimators: 20, maxDepth: 5)
        try await rf.fit(features: features, targets: targets)
        let preds = try await rf.predict(features: features)

        let correct = zip(targets.map { Int($0) }, preds).filter { $0 == $1 }.count
        let accuracy = Double(correct) / Double(targets.count)
        #expect(accuracy > 0.90)
    }

    @Test("RandomForestRegressor predicts linearly increasing values")
    func testRFRegression() async throws {
        let features: [[Double]] = (1...20).map { [Double($0)] }
        let targets: [Double]    = (1...20).map { Double($0) }

        let rf = try RandomForestRegressor(nEstimators: 20, maxDepth: 5)
        try await rf.fit(features: features, targets: targets)
        let preds = try await rf.predict(features: features)

        let mse = zip(targets, preds).map { pow($0 - $1, 2) }.reduce(0, +) / Double(targets.count)
        #expect(mse < 5.0)
    }

    @Test("RandomForestClassifier throws on invalid nEstimators")
    func testRFInvalidParams() {
        #expect(throws: MLError.self) {
            _ = try RandomForestClassifier(nEstimators: 0)
        }
    }

    @Test("mseImpurity calculates exact vector mean square variance")
    func testMSEImpurityCalculations() {
        #expect(mseImpurity([]) == 0.0)
        #expect(mseImpurity([5.0]) == 0.0)
        #expect(mseImpurity([3.0, 3.0, 3.0]) == 0.0)
        let vals = [1.0, 2.0, 3.0, 4.0]
        #expect(abs(mseImpurity(vals) - 1.25) < 1e-7)
    }
}

@Suite("GradientBoosting Tests")
struct GradientBoostingTests {

    @Test("GradientBoostedTreesRegressor fits and predicts a linear function")
    func testGBDTConvergence() async throws {
        let features: [[Double]] = (1...15).map { [Double($0)] }
        let targets: [Double]    = (1...15).map { Double($0) * 2.0 + 1.0 } // y = 2x + 1

        let gbdt = try GradientBoostedTreesRegressor(nEstimators: 50, learningRate: 0.3, maxDepth: 3)
        try await gbdt.fit(features: features, targets: targets)
        let preds = try await gbdt.predict(features: features)

        let mse = zip(targets, preds).map { pow($0 - $1, 2) }.reduce(0, +) / Double(targets.count)
        #expect(mse < 2.0)
    }

    @Test("GradientBoostedTreesRegressor throws when not fitted")
    func testGBDTNotFitted() async throws {
        let gbdt = try GradientBoostedTreesRegressor(nEstimators: 5)
        await #expect(throws: MLError.self) {
            _ = try await gbdt.predict(features: [[1.0]])
        }
    }

    @Test("GradientBoostedTreesRegressor quantile regression (10th, 50th, and 90th percentiles)")
    func testGBDTQuantileRegression() async throws {
        let features: [[Double]] = (1...30).map { [Double($0)] }
        // y = 2x + noise
        let targets: [Double] = (1...30).map { Double($0) * 2.0 + ($0 % 2 == 0 ? 5.0 : -5.0) }

        let q10 = try GradientBoostedTreesRegressor(nEstimators: 40, learningRate: 0.1, maxDepth: 2, loss: .quantile(alpha: 0.1))
        let q50 = try GradientBoostedTreesRegressor(nEstimators: 40, learningRate: 0.1, maxDepth: 2, loss: .quantile(alpha: 0.5))
        let q90 = try GradientBoostedTreesRegressor(nEstimators: 40, learningRate: 0.1, maxDepth: 2, loss: .quantile(alpha: 0.9))

        try await q10.fit(features: features, targets: targets)
        try await q50.fit(features: features, targets: targets)
        try await q90.fit(features: features, targets: targets)

        let testFeatures = [[10.0], [20.0]]
        let pred10 = try await q10.predict(features: testFeatures)
        let pred50 = try await q50.predict(features: testFeatures)
        let pred90 = try await q90.predict(features: testFeatures)

        // Monotonic ordering of quantiles: q10 <= q50 <= q90
        #expect(pred10[0] <= pred50[0])
        #expect(pred50[0] <= pred90[0])
        #expect(pred10[1] <= pred50[1])
        #expect(pred50[1] <= pred90[1])
    }
}
