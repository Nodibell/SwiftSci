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

        let rf = RandomForestClassifier(nEstimators: 10, maxDepth: 3)
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
        await #expect(throws: SwiftMLError.self) {
            try await tree.fit(features: [], targets: [])
        }
    }

    @Test("DecisionTreeClassifier throws when not fitted")
    func testClassifierNotFitted() async throws {
        let tree = DecisionTreeClassifier()
        await #expect(throws: SwiftMLError.self) {
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
        await #expect(throws: SwiftMLError.self) {
            _ = try await tree.predict(features: [[1.0]])
        }
    }

    // MARK: - Shape, Label & Finiteness Validation

    @Test("DecisionTree rejects ragged feature rows")
    func testRaggedFeatureMatrix() async throws {
        let raggedX = [
            [1.0, 2.0],
            [3.0],
            [4.0, 5.0]
        ]
        let y = [0.0, 1.0, 0.0]

        let clf = DecisionTreeClassifier()
        await #expect(throws: SwiftMLError.self) {
            try await clf.fit(features: raggedX, targets: y)
        }

        let reg = DecisionTreeRegressor()
        await #expect(throws: SwiftMLError.self) {
            try await reg.fit(features: raggedX, targets: y)
        }

        // Also test ragged row in predict
        let validX = [[1.0, 2.0], [3.0, 4.0]]
        let validY = [0.0, 1.0]
        try await clf.fit(features: validX, targets: validY)
        await #expect(throws: SwiftMLError.self) {
            _ = try await clf.predict(features: [[1.0]])
        }
    }

    @Test("predictProbability returns calibrated empirical probabilities with non-standard labels")
    func testEmpiricalClassProbabilitiesAndNegativeLabels() async throws {
        // Labels: -1.0, 10.0, 50.0 (arbitrary, non-contiguous, negative)
        // 10 samples: at a leaf of depth 1 or shallow tree with mixed samples
        let X: [[Double]] = [
            [1.0], [1.1], [1.2], [1.3], [1.4], [1.5], [1.6], [1.7], [1.8], [1.9]
        ]
        // 7 samples of class -1.0, 3 samples of class 50.0 (all at the same node with maxDepth: 0)
        let y: [Double] = [
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 50.0, 50.0, 50.0
        ]

        let tree = DecisionTreeClassifier(maxDepth: 0)
        try await tree.fit(features: X, targets: y)

        let classes = await tree.classes
        #expect(classes == [-1.0, 50.0])

        let probs = try await tree.predictProbability(features: [[1.2]])
        #expect(probs.count == 1)
        #expect(probs[0].count == 2)
        // Empirical distribution: 70% class -1.0, 30% class 50.0
        #expect(abs(probs[0][0] - 0.7) < 1e-6)
        #expect(abs(probs[0][1] - 0.3) < 1e-6)
    }

    @Test("DecisionTree rejects NaN in features and targets")
    func testNaNRejection() async throws {
        let nanFeatures = [[1.0, Double.nan], [2.0, 3.0]]
        let cleanFeatures = [[1.0, 2.0], [2.0, 3.0]]
        let nanTargets = [0.0, Double.nan]
        let cleanTargets = [0.0, 1.0]

        let clf = DecisionTreeClassifier()
        await #expect(throws: SwiftMLError.self) {
            try await clf.fit(features: nanFeatures, targets: cleanTargets)
        }
        await #expect(throws: SwiftMLError.self) {
            try await clf.fit(features: cleanFeatures, targets: nanTargets)
        }

        try await clf.fit(features: cleanFeatures, targets: cleanTargets)
        await #expect(throws: SwiftMLError.self) {
            _ = try await clf.predict(features: [[Double.nan, 2.0]])
        }
    }

    @Test("DecisionTree threshold calculation avoids overflow with large magnitude numbers")
    func testLargeMagnitudeThreshold() async throws {
        // Values near Double max that would overflow (a + b) to infinity
        let X = [
            [1.0e308],
            [1.5e308]
        ]
        let y = [0.0, 1.0]

        let tree = DecisionTreeClassifier(maxDepth: 1)
        try await tree.fit(features: X, targets: y)

        let nodes = await tree.flatNodes
        #expect(!nodes.isEmpty)
        let root = nodes[0]
        #expect(!root.isLeaf)
        #expect(root.threshold.isFinite)
        #expect(abs(root.threshold - 1.25e308) / 1.25e308 < 1e-6)

        let pred1 = try await tree.predict(features: [[1.0e308]])
        let pred2 = try await tree.predict(features: [[1.5e308]])
        #expect(pred1 == [0])
        #expect(pred2 == [1])
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

        let rf = RandomForestClassifier(nEstimators: 20, maxDepth: 5)
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

        let rf = RandomForestRegressor(nEstimators: 20, maxDepth: 5)
        try await rf.fit(features: features, targets: targets)
        let preds = try await rf.predict(features: features)

        let mse = zip(targets, preds).map { pow($0 - $1, 2) }.reduce(0, +) / Double(targets.count)
        #expect(mse < 5.0)
    }

    @Test("RandomForestClassifier throws on invalid nEstimators")
    func testRFInvalidParams() async {
        let rf = RandomForestClassifier(nEstimators: 0)
        await #expect(throws: SwiftMLError.self) {
            try await rf.fit(features: [[1.0]], targets: [0.0])
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

    @Test("RandomForestClassifier is deterministic with identical randomState")
    func testRFDeterminism() async throws {
        var features = [[Double]]()
        var targets = [Double]()
        for i in 0..<40 {
            features.append([Double(i), Double(i % 3)])
            targets.append(Double(i % 2))
        }

        let rf1 = RandomForestClassifier(nEstimators: 10, maxDepth: 4, randomState: 12345)
        try await rf1.fit(features: features, targets: targets)
        let preds1 = try await rf1.predict(features: features)

        let rf2 = RandomForestClassifier(nEstimators: 10, maxDepth: 4, randomState: 12345)
        try await rf2.fit(features: features, targets: targets)
        let preds2 = try await rf2.predict(features: features)

        #expect(preds1 == preds2)
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
        await #expect(throws: SwiftMLError.self) {
            _ = try await gbdt.predict(features: [[1.0]])
        }
    }

    @Test("GradientBoostedTreesRegressor quantile regression (10th, 50th, and 90th percentiles)")
    func testGBDTQuantileRegression() async throws {
        let features: [[Double]] = (1...30).map { [Double($0)] }
        var targets: [Double] = []
        targets.reserveCapacity(30)
        for i in 1...30 {
            let base: Double = Double(i) * 2.0
            let noise: Double = (i % 2 == 0) ? 5.0 : -5.0
            targets.append(base + noise)
        }

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

    // MARK: - Loss, Early Stopping & Hyperparameter Validation

    @Test("Absolute error loss resists extreme outliers via median leaf adjustments")
    func testAbsoluteErrorOutlierResistance() async throws {
        // Linear data y = 3 * x with a severe outlier at x = 5
        var features: [[Double]] = []
        var targets: [Double] = []
        for i in 1...20 {
            let x = Double(i)
            features.append([x])
            if i == 5 {
                targets.append(1000.0) // Severe outlier
            } else {
                targets.append(3.0 * x)
            }
        }

        let l1Model = try GradientBoostedTreesRegressor(nEstimators: 60, learningRate: 0.1, maxDepth: 3, loss: .absoluteError)
        try await l1Model.fit(features: features, targets: targets)

        // Evaluate on clean points away from outlier
        let testPoints = [[2.0], [10.0], [18.0]]
        let l1Preds = try await l1Model.predict(features: testPoints)

        // Expected roughly 6, 30, 54
        #expect(abs(l1Preds[0] - 6.0) < 3.0)
        #expect(abs(l1Preds[1] - 30.0) < 5.0)
        #expect(abs(l1Preds[2] - 54.0) < 5.0)
    }

    @Test("Early stopping monitors loss-specific metric")
    func testLossSpecificEarlyStopping() async throws {
        let trainX = (1...20).map { [Double($0)] }
        let trainY = (1...20).map { Double($0) * 2.0 }
        let valX = (21...25).map { [Double($0)] }
        let valY = (21...25).map { Double($0) * 2.0 }

        let earlyStopping = EarlyStopping(patience: 3, minDelta: 0.001)
        let model = try GradientBoostedTreesRegressor(nEstimators: 100, learningRate: 0.1, maxDepth: 2, loss: .absoluteError)
        try await model.fit(
            features: trainX,
            targets: trainY,
            validationFeatures: valX,
            validationTargets: valY,
            earlyStopping: earlyStopping
        )

        let trees = await model.getEnsembleTrees()
        // Should have converged/stopped well before 100 estimators
        #expect(trees.count < 100)
    }

    @Test("Validation data dimension mismatch and NaN rejection")
    func testValidationDataValidation() async throws {
        let trainX = [[1.0, 2.0], [3.0, 4.0]]
        let trainY = [1.0, 2.0]

        let model = try GradientBoostedTreesRegressor(nEstimators: 5)
        // Mismatched feature width in validation
        await #expect(throws: SwiftMLError.self) {
            try await model.fit(
                features: trainX,
                targets: trainY,
                validationFeatures: [[1.0]],
                validationTargets: [1.0]
            )
        }

        // Mismatched count in validation features vs targets
        await #expect(throws: SwiftMLError.self) {
            try await model.fit(
                features: trainX,
                targets: trainY,
                validationFeatures: [[1.0, 2.0]],
                validationTargets: [1.0, 2.0]
            )
        }

        // NaN in validation targets
        await #expect(throws: SwiftMLError.self) {
            try await model.fit(
                features: trainX,
                targets: trainY,
                validationFeatures: [[1.0, 2.0]],
                validationTargets: [Double.nan]
            )
        }
    }

    @Test("predict validates feature dimensions and rejects NaN")
    func testPredictValidation() async throws {
        let trainX = [[1.0, 2.0], [3.0, 4.0]]
        let trainY = [5.0, 7.0]

        let model = try GradientBoostedTreesRegressor(nEstimators: 5)
        try await model.fit(features: trainX, targets: trainY)

        // Wrong number of features in prediction
        await #expect(throws: SwiftMLError.self) {
            _ = try await model.predict(features: [[1.0]])
        }

        // NaN in prediction features
        await #expect(throws: SwiftMLError.self) {
            _ = try await model.predict(features: [[1.0, Double.nan]])
        }
    }

    @Test("Hyperparameter validation in GradientBoostedTreesRegressor")
    func testHyperparameterValidation() {
        #expect(throws: SwiftMLError.self) {
            _ = try GradientBoostedTreesRegressor(nEstimators: 0)
        }
        #expect(throws: SwiftMLError.self) {
            _ = try GradientBoostedTreesRegressor(learningRate: 0.0)
        }
        #expect(throws: SwiftMLError.self) {
            _ = try GradientBoostedTreesRegressor(learningRate: Double.infinity)
        }
        #expect(throws: SwiftMLError.self) {
            _ = try GradientBoostedTreesRegressor(maxDepth: 0)
        }
        #expect(throws: SwiftMLError.self) {
            _ = try GradientBoostedTreesRegressor(minSamplesSplit: 1)
        }
        #expect(throws: SwiftMLError.self) {
            _ = try GradientBoostedTreesRegressor(loss: .quantile(alpha: 0.0))
        }
        #expect(throws: SwiftMLError.self) {
            _ = try GradientBoostedTreesRegressor(loss: .quantile(alpha: 1.0))
        }
        #expect(throws: SwiftMLError.self) {
            _ = try GradientBoostedTreesRegressor(loss: .quantile(alpha: Double.nan))
        }
    }
}
