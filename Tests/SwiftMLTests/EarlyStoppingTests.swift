import Testing
import Foundation
@testable import SwiftML

@Suite("EarlyStopping Callback Tests")
struct EarlyStoppingTests {

    @Test("EarlyStopping state transitions for minimization")
    func testEarlyStoppingMinimization() {
        var es = EarlyStopping(patience: 3, minDelta: 0.1, direction: .minimize, restoreBestWeights: true)

        // Epoch 0: score 10.0 -> new best
        let s0 = es.step(currentScore: 10.0, epoch: 0)
        #expect(s0.isBest == true)
        #expect(s0.shouldStop == false)
        #expect(es.wait == 0)
        #expect(es.bestScore == 10.0)

        // Epoch 1: score 9.8 -> improvement is 0.2 >= 0.1 -> new best
        let s1 = es.step(currentScore: 9.8, epoch: 1)
        #expect(s1.isBest == true)
        #expect(s1.shouldStop == false)
        #expect(es.wait == 0)

        // Epoch 2: score 9.75 -> improvement is 0.05 < 0.1 -> not enough, wait = 1
        let s2 = es.step(currentScore: 9.75, epoch: 2)
        #expect(s2.isBest == false)
        #expect(s2.shouldStop == false)
        #expect(es.wait == 1)

        // Epoch 3: score 9.9 -> worse, wait = 2
        let s3 = es.step(currentScore: 9.9, epoch: 3)
        #expect(s3.isBest == false)
        #expect(s3.shouldStop == false)
        #expect(es.wait == 2)

        // Epoch 4: score 10.5 -> worse, wait = 3 >= patience (3) -> shouldStop = true
        let s4 = es.step(currentScore: 10.5, epoch: 4)
        #expect(s4.isBest == false)
        #expect(s4.shouldStop == true)
        #expect(es.bestEpoch == 1)
    }

    @Test("EarlyStopping state transitions for maximization")
    func testEarlyStoppingMaximization() {
        var es = EarlyStopping(patience: 2, minDelta: 0.05, direction: .maximize)

        _ = es.step(currentScore: 0.70, epoch: 0)
        _ = es.step(currentScore: 0.76, epoch: 1) // +0.06 -> best
        #expect(es.bestScore == 0.76)

        _ = es.step(currentScore: 0.78, epoch: 2) // +0.02 < 0.05 -> wait = 1
        #expect(es.wait == 1)

        let s3 = es.step(currentScore: 0.77, epoch: 3) // wait = 2 -> shouldStop
        #expect(s3.shouldStop == true)
    }

    @Test("GradientBoostedTreesRegressor stops early on plateau")
    func testGBDTEarlyStopping() async throws {
        let es = EarlyStopping(patience: 3, minDelta: 0.01, direction: .minimize, restoreBestWeights: true)
        let reg = try GradientBoostedTreesRegressor(nEstimators: 50, learningRate: 0.1, maxDepth: 3)

        // Generate synthetic linear data
        var X: [[Double]] = []
        var y: [Double] = []
        for i in 0..<30 {
            let v = Double(i)
            X.append([v, v * 0.5])
            y.append(v * 2.0)
        }

        // Validation set is same linear function
        let valX: [[Double]] = [[1.0, 0.5], [5.0, 2.5], [10.0, 5.0]]
        let valY: [Double] = [2.0, 10.0, 20.0]

        try await reg.fit(
            features: X,
            targets: y,
            validationFeatures: valX,
            validationTargets: valY,
            earlyStopping: es
        )

        let preds = try await reg.predict(features: [[2.0, 1.0]])
        #expect(preds.count == 1)
        #expect(abs(preds[0] - 4.0) < 1.0)
    }

    @Test("MLPClassifier early stopping execution")
    func testMLPEarlyStopping() async throws {
        let es = EarlyStopping(patience: 4, minDelta: 1e-4, direction: .minimize, restoreBestWeights: true)
        let mlp = MLPClassifier(hiddenLayerSizes: [8, 4], maxIter: 100, learningRate: 0.05)

        let X: [[Double]] = [
            [-1.0, -1.0], [-1.2, -0.8], [-0.9, -1.1],
            [1.0, 1.0], [1.2, 0.8], [0.9, 1.1]
        ]
        let y: [Double] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

        let valX: [[Double]] = [[-1.1, -1.0], [1.1, 1.0]]
        let valY: [Double] = [0.0, 1.0]

        try await mlp.fit(
            features: X,
            targets: y,
            validationFeatures: valX,
            validationTargets: valY,
            earlyStopping: es
        )

        let preds = try await mlp.predict(features: [[-1.0, -1.0], [1.0, 1.0]])
        #expect(preds == [0, 1])
    }
}
